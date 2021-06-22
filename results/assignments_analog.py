from ytoloka import YToloka, TYPE_ANALOG

POOL_TYPE = TYPE_ANALOG
RESULTS_PATH = 'image-assignments-analog.csv'


def run():
    ytoloka = YToloka()
    ytoloka.get_or_create_pool(POOL_TYPE)
    ytoloka.upload_tasks_analog()
    ytoloka.open_pool(POOL_TYPE)


def get_results():
    ytoloka = YToloka()
    df = ytoloka._get_assignments_raw(POOL_TYPE)
    df.to_csv(RESULTS_PATH, index=False)


if __name__ == '__main__':
    run()
