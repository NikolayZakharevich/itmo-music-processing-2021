import datetime
import re
from typing import List, Union

import pandas as pd
from dateutil.utils import today
from numpy import array_split
from toloka.client import TolokaClient
from toloka.client.actions import RestrictionV2
from toloka.client.batch_create_results import TaskSuiteBatchCreateResult
from toloka.client.collectors import Income, SkippedInRowAssignments, MajorityVote
from toloka.client.conditions import IncomeSumForLast24Hours, SkippedInRowCount, TotalAnswersCount, \
    CorrectAnswersRate
from toloka.client.filter import FilterAnd, FilterOr, RegionByPhone, Languages, \
    ClientType
from toloka.client.owner import Owner
from toloka.client.pool import Pool
from toloka.client.primitives.operators import InclusionOperator, IdentityOperator, CompareOperator
from toloka.client.quality_control import QualityControl
from toloka.client.task import BaseTask
from toloka.client.task_suite import TaskSuite
from toloka.client.user_restriction import UserRestriction, DurationUnit

from app.columns import Emotion, Column, AssignmentColumn
from config import TOKEN_YANDEX_TOLOKA, YANDEX_TOLOKA_PROJECT_ID, YANDEX_TOLOKA_TRAINING_POOL_ID, STORAGE_URL_STATIC

REWARD_PER_ASSIGNMENT = 0.01
TASKS_PER_TASK_SUITE = 20
TRAINING_PASSING_SKILL_VALUE = 90
OVERLAP = 3


# Conventions:
# 1. Pool private comment should be equal to emotion name
class YToloka:
    client: TolokaClient = None

    def get_or_create_pool(self, emotion: Emotion) -> Pool:
        self._check_auth()

        project_id = YANDEX_TOLOKA_PROJECT_ID
        emotion_str = emotion.value

        all_pools = self.client.get_pools(project_id=project_id)
        pool = next(filter(lambda p: p.private_comment == emotion_str, all_pools), None)

        if pool is None:
            pool = self.client.create_pool(self._get_pool(project_id, emotion_str))

        return pool

    def update_pool(self, emotion: Emotion):
        self._check_auth()
        self.client.update_pool(YANDEX_TOLOKA_PROJECT_ID, emotion.value)

    def upload_tasks(self, emotion: Emotion, tracks: pd.DataFrame) -> TaskSuiteBatchCreateResult:
        return self.upload_tasks_by_paths(emotion, tracks[Column.AUDIO_PATH.value])

    def upload_tasks_by_paths(self, emotion: Emotion, audio_paths: pd.Series) -> TaskSuiteBatchCreateResult:
        if len(audio_paths) == 0:
            return TaskSuiteBatchCreateResult()

        self._check_auth()

        emotion_str = emotion.value
        pool_id = self._get_pool_id(emotion)

        tasks: List[BaseTask] = []
        for audio_path in audio_paths.tolist():
            tasks.append(BaseTask(input_values={
                'emotion': emotion_str,
                Column.AUDIO_PATH.value: STORAGE_URL_STATIC + audio_path,
            }))

        task_suites: List[TaskSuite] = []
        n_suites = (len(tasks) + TASKS_PER_TASK_SUITE - 1) // TASKS_PER_TASK_SUITE
        for tasks_batch in array_split(tasks, n_suites):
            task_suites.append(TaskSuite(
                overlap=OVERLAP,
                pool_id=pool_id,
                tasks=tasks_batch.tolist()
            ))

        return self.client.create_task_suites(task_suites=task_suites)

    def stop_pool(self, emotion: Emotion):
        pool_id = self._get_pool_id(emotion)

        task_suites = self.client.get_task_suites(pool_id=pool_id)
        for task_suite in task_suites:
            if task_suite.remaining_overlap > 0:
                self.client.patch_task_suite_overlap_or_min(task_suite.id, overlap=0)

    def update_assignments(self, assignments: pd.DataFrame, emotion: Emotion) -> pd.DataFrame:
        updated_assignments = {}
        for _, assignment in assignments.iterrows():
            track_id = assignment[AssignmentColumn.TRACK_ID.value]
            emotion_str = assignment[AssignmentColumn.EMOTION.value]
            assignment_hash = f'{track_id}:{emotion_str}'
            updated_assignments[assignment_hash] = assignment

        assignments_raw = self._get_assignments_raw(emotion)
        assignments_mv = YToloka._apply_majority_vote(assignments_raw, emotion)
        for _, assignment in assignments_mv.iterrows():
            track_id = assignment[AssignmentColumn.TRACK_ID.value]
            emotion_str = assignment[AssignmentColumn.EMOTION.value]
            assignment_hash = f'{track_id}:{emotion_str}'
            updated_assignments[assignment_hash] = assignment

        return pd.DataFrame(data=updated_assignments.values(), columns=assignments.columns)

    def open_pool(self, emotion: Emotion):
        self.client.open_pool(self._get_pool_id(emotion))

    @staticmethod
    def update_emotions(tracks: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        approved_emotions = {}
        for _, assignment in assignments.iterrows():

            if assignment[AssignmentColumn.RATE.value] < 1.0:
                continue

            track_id = assignment[AssignmentColumn.TRACK_ID.value]
            emotion = assignment[AssignmentColumn.EMOTION.value]

            emotions = approved_emotions.get(track_id, [])
            emotions.append(emotion)
            approved_emotions[track_id] = emotions

        updated_tracks = []
        for _, track in tracks.iterrows():
            track_id = track[Column.YM_TRACK_ID.value]
            track[Column.EMOTIONS.value] = '|'.join(approved_emotions.get(track_id, []))
            updated_tracks.append(track)

        return pd.DataFrame(data=updated_tracks, columns=[c.value for c in Column])

    @staticmethod
    def filter_assigned(tracks: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        assigned_set = set(assignments[AssignmentColumn.TRACK_ID.value].tolist())

        unassigned_tracks = []
        for i, track in tracks.iterrows():
            if track[Column.YM_TRACK_ID.value] in assigned_set:
                unassigned_tracks.append(track)
        return pd.DataFrame(unassigned_tracks)

    def _get_assignments_raw(self, emotion: Emotion) -> pd.DataFrame:
        self._check_auth()

        pool_id = self._get_pool_id(emotion)
        return self.client.get_assignments_df(
            pool_id=pool_id,
            field=[]
        )

    @staticmethod
    def _apply_majority_vote(assignments_raw: pd.DataFrame, emotion: Emotion) -> pd.DataFrame:
        column_raw_input = f'INPUT:{Column.AUDIO_PATH.value}'
        column_raw_output = 'OUTPUT:result'

        column_track_id = AssignmentColumn.TRACK_ID.value
        column_rate = AssignmentColumn.RATE.value
        column_yes = AssignmentColumn.YES.value
        column_no = AssignmentColumn.NO.value

        def extract_track_id(audio_url) -> Union[str, None]:
            credentials_search_v1 = re.search(fr'{STORAGE_URL_STATIC}.+/(\d+):(\d+).mp3', audio_url)
            if credentials_search_v1:
                track_id = credentials_search_v1.group(2)
                return track_id
            credentials_search_v2 = re.search(fr'{STORAGE_URL_STATIC}.+/(\d+).mp3', audio_url)
            if credentials_search_v2:
                track_id = credentials_search_v2.group(1)
                return track_id
            return None

        assignments = assignments_raw \
            .groupby([column_raw_input, column_raw_output]) \
            .size() \
            .unstack(fill_value=0) \
            .reset_index()

        assignments[column_track_id] = assignments[column_raw_input].apply(extract_track_id)
        assignments[column_rate] = assignments[column_yes] / (assignments[column_yes] + assignments[column_no])
        assignments[AssignmentColumn.EMOTION.value] = emotion.value

        return assignments[[c.value for c in AssignmentColumn]] \
            .sort_values(AssignmentColumn.RATE.value, ascending=False)

    def _check_auth(self):
        if self.client is None:
            self.client = TolokaClient(TOKEN_YANDEX_TOLOKA, 'PRODUCTION')

    @staticmethod
    def _get_pool(project_id: str, emotion_str: str) -> Pool:
        return Pool(
            project_id=project_id,
            private_name=f'Верно ли указана эмоция трека? [{emotion_str}]',
            may_contain_adult_content=False,
            reward_per_assignment=REWARD_PER_ASSIGNMENT,
            assignment_max_duration_seconds=600,
            defaults=Pool.Defaults(default_overlap_for_new_task_suites=OVERLAP),
            will_expire=today() + datetime.timedelta(days=365),
            private_comment=emotion_str,
            auto_close_after_complete_delay_seconds=0,
            auto_accept_solutions=True,
            auto_accept_period_day=21,
            assignments_issuing_config=Pool.AssignmentsIssuingConfig(issue_task_suites_in_creation_order=False),
            priority=0,
            filter=FilterAnd(
                and_=[
                    FilterOr(
                        or_=[
                            RegionByPhone(InclusionOperator.IN, 225),
                            RegionByPhone(InclusionOperator.IN, 187),
                            RegionByPhone(InclusionOperator.IN, 159),
                            RegionByPhone(InclusionOperator.IN, 149),
                        ]
                    ),
                    FilterOr(
                        or_=[
                            Languages(InclusionOperator.IN, 'RU')
                        ]
                    ),
                    FilterOr(
                        or_=[
                            ClientType(IdentityOperator.EQ, ClientType.ClientType.TOLOKA_APP),
                            ClientType(IdentityOperator.EQ, ClientType.ClientType.BROWSER),
                        ]
                    )
                ]
            ),
            quality_control=QualityControl(
                training_requirement=QualityControl.TrainingRequirement(
                    training_passing_skill_value=TRAINING_PASSING_SKILL_VALUE,
                    training_pool_id=YANDEX_TOLOKA_TRAINING_POOL_ID,
                ),
                configs=[
                    QualityControl.QualityControlConfig(
                        collector_config=Income(),
                        rules=[
                            QualityControl.QualityControlConfig.RuleConfig(
                                action=RestrictionV2(
                                    duration=1,
                                    scope=UserRestriction.Scope.ALL_PROJECTS,
                                    duration_unit=DurationUnit.DAYS
                                ),
                                conditions=[
                                    IncomeSumForLast24Hours(CompareOperator.GTE, 20.0)
                                ]
                            )
                        ]
                    ),
                    QualityControl.QualityControlConfig(
                        collector_config=SkippedInRowAssignments(),
                        rules=[
                            QualityControl.QualityControlConfig.RuleConfig(
                                action=RestrictionV2(
                                    duration=1,
                                    scope=UserRestriction.Scope.PROJECT,
                                    duration_unit=DurationUnit.DAYS
                                ),
                                conditions=[
                                    SkippedInRowCount(CompareOperator.GTE, 10)
                                ]
                            )
                        ]
                    ),
                    QualityControl.QualityControlConfig(
                        collector_config=MajorityVote(answer_threshold=3, history_size=10),
                        rules=[
                            QualityControl.QualityControlConfig.RuleConfig(
                                action=RestrictionV2(
                                    duration=1,
                                    scope=UserRestriction.Scope.PROJECT,
                                    duration_unit=DurationUnit.DAYS
                                ),
                                conditions=[
                                    TotalAnswersCount(CompareOperator.GTE, 4),
                                    CorrectAnswersRate(CompareOperator.LT, 75)
                                ]
                            )
                        ]
                    )
                ]
            ),
            owner=Owner(
                id='27fe95ef057e6e64532525a578c472f9',
                myself=True
            ),
            status=Pool.Status.CLOSED,
            type=Pool.Type.REGULAR
        )

    def _get_pool_id(self, emotion: Emotion) -> str:
        return self.get_or_create_pool(emotion).id