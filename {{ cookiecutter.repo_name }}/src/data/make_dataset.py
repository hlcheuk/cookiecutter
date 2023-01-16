import pandas_gbq
from src.utils.logger import setup_logger

__all__ = "Make_dataset"
logger = setup_logger(__all__)


def bq_query_path(query_path: str,
                  start_date: str = None,
                  end_date: str = None,
                  project: str = 'hktv-data'):
    """
    Wrapper for BQ client function

    Args:
        query (str): path of a SQL query
        start_date (str): start date of the targeted period in format of 'YYYYMMDD'
        end_date (str): end date of the targeted period in format of 'YYYYMMDD'
        project (str): gcp project name. e.g. hktv-data

    Example:
        project_id = "hktv-data"
        sql = "./src/sql/category_name.sql"
        df = bq_query(project_id, sql)

    Returns:
        df: Query result in dataframe format

    TODO:
        (Maybe) Use service account json instead of bigquery.Client for authentication
    """
    # Open the SQL script and load as string
    sql_file = open(query_path, 'r')
    query = sql_file.read()
    logger.info(f"Read from the specified SQL - {query_path}")
    sql_file.close()
    # added period to the query if provided
    if start_date is not None and end_date is not None:
        query = query.format(START_DATE=start_date, END_DATE=end_date)
    elif start_date is None and end_date is not None:
        raise KeyError('Please provide start date')
    elif start_date is not None and end_date is None:
        raise KeyError('Please provide end date')
    # read the query from BQ
    df = pandas_gbq.read_gbq(query, project_id=project)
    logger.info(f"Downloaded from BQ for - {df.shape[0]} records")

    return df


if __name__ == "__main__":
    bq_query_path()