import datetime
import random
from typing import List

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

from config import TOKEN_YANDEX_TOLOKA, YANDEX_TOLOKA_PROJECT_ID

REWARD_PER_ASSIGNMENT = 0.01
TASKS_PER_TASK_SUITE = 20
TRAINING_PASSING_SKILL_VALUE = 90
OVERLAP = 20

TYPE_ORIGINAL = 'original'
TYPE_ANALOG = 'analog'
TYPE_RANDOM = 'random'


class YToloka:
    client: TolokaClient = None

    def get_or_create_pool(self, pool_type: str) -> Pool:
        self._check_auth()

        project_id = YANDEX_TOLOKA_PROJECT_ID

        all_pools = self.client.get_pools(project_id=project_id)
        pool = next(filter(lambda p: p.private_comment == pool_type, all_pools), None)

        if pool is None:
            pool = self.client.create_pool(self._get_pool(project_id, pool_type))

        return pool

    def upload_tasks(self, pool_type: str, tasks: List[BaseTask]) -> TaskSuiteBatchCreateResult:
        self._check_auth()
        pool_id = self._get_pool_id(pool_type)

        task_suites: List[TaskSuite] = []
        n_suites = (len(tasks) + TASKS_PER_TASK_SUITE - 1) // TASKS_PER_TASK_SUITE
        for tasks_batch in array_split(tasks, n_suites):
            task_suites.append(TaskSuite(
                overlap=OVERLAP,
                pool_id=pool_id,
                tasks=tasks_batch.tolist()
            ))

        return self.client.create_task_suites(task_suites=task_suites)

    def upload_tasks_analog(self) -> TaskSuiteBatchCreateResult:
        df = pd.read_csv('my_vs_another.csv')
        if len(df) == 0:
            return TaskSuiteBatchCreateResult()

        tasks: List[BaseTask] = []
        for _, row in df.iterrows():
            tasks.append(BaseTask(input_values={
                'image_first': row['cover_generated_my'],
                'image_second': row['cover_generated_another'],
                'audio_path': row['audio']
            }))
        return self.upload_tasks(TYPE_ANALOG, tasks)

    def upload_tasks_original(self) -> TaskSuiteBatchCreateResult:
        df = pd.read_csv('my_vs_original.csv')
        if len(df) == 0:
            return TaskSuiteBatchCreateResult()

        tasks: List[BaseTask] = []
        for _, row in df.iterrows():
            tasks.append(BaseTask(input_values={
                'image_first': row['cover_original'],
                'image_second': row['cover_generated'],
                'audio_path': row['audio']
            }))
        return self.upload_tasks(TYPE_ORIGINAL, tasks)

    def upload_tasks_random(self) -> TaskSuiteBatchCreateResult:
        df = pd.read_csv('my_vs_original.csv')
        if len(df) == 0:
            return TaskSuiteBatchCreateResult()

        covers = df.cover_generated.tolist()

        tasks: List[BaseTask] = []
        for _, row in df.iterrows():
            tasks.append(BaseTask(input_values={
                'image_first': row['cover_generated'],
                'image_second': random.choice(covers),
                'audio_path': row['audio']
            }))
        return self.upload_tasks(TYPE_RANDOM, tasks)

    def stop_pool(self, pool_type: str):
        pool_id = self._get_pool_id(pool_type)

        task_suites = self.client.get_task_suites(pool_id=pool_id)
        for task_suite in task_suites:
            if task_suite.remaining_overlap > 0:
                self.client.patch_task_suite_overlap_or_min(task_suite.id, overlap=0)

    def open_pool(self, pool_type: str):
        try:
            self.client.open_pool(self._get_pool_id(pool_type))
        except Exception as e:
            print(e)
            pass

    @staticmethod
    def filter_assigned(tracks: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        assigned_set = set(assignments['track_id'].tolist())

        unassigned_tracks = []
        for i, track in tracks.iterrows():
            if track['track_id'] in assigned_set:
                unassigned_tracks.append(track)
        return pd.DataFrame(unassigned_tracks)

    def _get_assignments_raw(self, pool_type: str) -> pd.DataFrame:
        self._check_auth()

        pool_id = self._get_pool_id(pool_type)
        return self.client.get_assignments_df(
            pool_id=pool_id,
            field=[]
        )

    def _check_auth(self):
        if self.client is None:
            self.client = TolokaClient(TOKEN_YANDEX_TOLOKA, 'PRODUCTION')

    @staticmethod
    def _get_pool(project_id: str, pool_type: str) -> Pool:
        return Pool(
            project_id=project_id,
            private_name=f'Какая обложка подходит лучше? [{pool_type}]',
            may_contain_adult_content=False,
            reward_per_assignment=REWARD_PER_ASSIGNMENT,
            assignment_max_duration_seconds=600,
            defaults=Pool.Defaults(default_overlap_for_new_task_suites=OVERLAP),
            will_expire=today() + datetime.timedelta(days=365),
            private_comment=pool_type,
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
                # training_requirement=QualityControl.TrainingRequirement(
                #     training_passing_skill_value=TRAINING_PASSING_SKILL_VALUE,
                #     training_pool_id=YANDEX_TOLOKA_TRAINING_POOL_ID,
                # ),
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

    def _get_pool_id(self, pool_type: str) -> str:
        return self.get_or_create_pool(pool_type).id
