from ytoloka import YToloka, TYPE_ORIGINAL

POOL_TYPE = TYPE_ORIGINAL
RESULTS_PATH = 'image-assignments-original.csv'


def run():
    ytoloka = YToloka()
    ytoloka.get_or_create_pool(POOL_TYPE)
    ytoloka.upload_tasks_original()
    ytoloka.open_pool(POOL_TYPE)


def get_results():
    ytoloka = YToloka()
    df = ytoloka._get_assignments_raw(POOL_TYPE)
    df.to_csv(RESULTS_PATH, index=False)


if __name__ == '__main__':
    run()
    # get_results()
