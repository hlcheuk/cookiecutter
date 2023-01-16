import os
from collections import Counter
import numpy as np
import pandas as pd
import pandas_gbq
from google.cloud import storage, bigquery
from google.oauth2 import service_account
import google.auth

from src.utils.utils import deprecated
from src.utils.logger import setup_logger

__all__ = "Gcp_utils"
logger = setup_logger(__all__)


def read_gs(sheet_id: str, sheet_name: str, cred: str = None):
    """To read data from a google spreadsheet as a pandas DataFrame

    Args:
        sheet_id (str): the spreadsheet ID which can be obtained from the url of the google sheet
        sheet_name (str): the name of the worksheet of the spreadsheet
        cred (str): the path of the secret key JSON file

    Returns:
        pd.DataFrame: data from the specified sheet of the spreadsheet

    TODO:
        enable oauth for authentication
    """
    # Import required package
    import gspread

    # Get the path of the service account key
    if cred == None:
        cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred == None:
            raise OSError(
                "Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set."
            )

    # Authentication with a service account key file
    gc = gspread.service_account(filename=cred)

    # Open the specified spreadsheet
    try:
        spr = gc.open_by_key(sheet_id)
    except gspread.exceptions.SpreadsheetNotFound as e:
        print("Spreadsheet not found.")
        print(
            f"Please check the spreadsheet ID\nor if it is shared with the service account.\n {e}"
        )
        exit(1)
    # Open the specified worksheet
    try:
        wks = spr.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound as e:
        print(f"Worksheet '{e}' not found.")
        print(
            f"Please check the worksheet name or if it is accessible by the service account."
        )
        exit(1)

    # Get a list of rows which contain the data in the worksheet
    rows = wks.get_all_values()
    col = rows[0]

    # Check if the data in the spreadsheet starts at Cell A1
    if set(col) == {''}:
        raise ValueError("Data as a table should be started at cell 'A1'")
    # Check if there are any unnamed columns
    if "" in col:
        raise ValueError("There are unnamed column(s)")

    # Reformat the column names and check for duplicates
    col = [c.lower().replace(' ', '_') for c in col]
    if len(col) != len(set(col)):
        dup_col = [key for key, val in Counter(col).items() if val > 1]
        raise ValueError(f"There are columns with duplicated names: {dup_col}")

    # Convert the list to a DataFrame and render
    df = pd.DataFrame.from_records(rows[1:(len(rows))], columns=col)

    return df


def generate_schema(df: pd.DataFrame):
    """
    This function automatically detect and output the schema of a dataframe
    to a ordered dictionary which can be further loaded into the process of
    uploading dataframe (especially the ones having nested fields) to GCP Bigquery

    Argument:
        df: A pandas dataframe

    Return:
        a list of ordered dictionaries which describes the schema of the dataframe

    Example 1:
        df = pd.DataFrame({'list': [[1, 2, 3]],                                     # array
                      'dict': [{'key1': 'value1'}],                                 # struct
                      'list_of_dict': [[{'key1': 'value1'}, {'key2': 'value2'}]]})  # array of structs
        generate_schema(df)

    Example 2:
        df = pd.DataFrame({'wine_id': ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'],
                   'region': ['Bordeaux', 'Burgundy', 'Chateauneuf du Pape', 'Bordeaux', 'Loire Valley', 'Burgundy'],
                   'price': [120, 130, 260, 220, 230, 210],
                   'vintage': ['2018', '2019', '2015', '2015', '2017', '2014']})

        df_list = df.groupby('region', as_index=False).agg({'wine_id': lambda srs: list(srs)})
        df_dict = df.assign(vintage_price = lambda df: df.filter(['vintage', 'price']).to_dict(orient='records')).filter(['region', 'vintage_price'])
        df_list_of_dict = df_dict.groupby('region', as_index=False).agg({'vintage_price': lambda srs: list(srs)})

        generate_schema(df_list), generate_schema(df_dict), generate_schema(df_list_of_dict)
    """
    # import required packages
    from bigquery_schema_generator.generate_schema import SchemaGenerator
    json_text = df.to_dict("records")

    # assuming there is no table created before and initialise the schema
    schema_map = {}

    generator = SchemaGenerator(input_format="dict",
                                quoted_values_are_strings=True,
                                keep_nulls=True)
    schema_map, error_logs = generator.deduce_schema(input_data=json_text,
                                                     schema_map=schema_map)
    schema = generator.flatten_schema(schema_map)

    return schema


def create_bq_table(project_id: str, dataset_id: str, table_id: str,
                    schema: list, client):
    """
    To create a bigquery table

    Argument:
        dataset_id = the dataset which the table to be located
        table_id = the name of the table
        schema = schema generated from the dataframe which is to be uploaded
        client = bigquery client

    Usage:
        df = pd.DataFrame({'list': [[1, 2, 3]],
                      'dict': [{'key1': 'value1'}],
                      'list_of_dict': [[{'key1': 'A', 'key2': 1}, {'key1': 'B', 'key2': 2}, {'key1': 'C', 'key2': 3}]]})

        create_bq_table('hktv-data',
            'working_data',
            'trial',
            generate_schema(df),
            bigquery.Client())

    TODO:
        raise error for invalid field name
        raise error for already exist table
    """
    table = project_id + "." + dataset_id + "." + table_id
    bq_table = bigquery.Table(table, schema=schema)

    # With exists_ok = True, ignore "already exists" errors when creating the table
    client.create_table(bq_table)
    # logger.info("table created whether it already exists or not.")


def upload_df_to_bq(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    method: str,
    cred: str = None,
):
    """
    Wrapper of the to_gbq() method of class pd.DataFrame
    To upload a pandas dataframe to GCP Bigquery. The upload data can contain a nested column (array/ array of struct)

    Reference:
        https://cloud.google.com/bigquery/docs/pandas-gbq-migration
        https://medium.com/inside-league/loading-complex-json-files-in-realtime-to-bigquery-from-pubsub-using-dataflow-and-updating-the-49b420838bac

    Args:
        df (pd.DataFrame) = the target dataframe to upload
        project_id (str) = the project ID where the dataframe is going to be uploaded
        dataset_id (str) = the dataset ID where the dataframe is going to be uploaded
        table_id (str) = the table ID where the dataframe is going to be stored
        method (str) = the method to upload the data if the destinated table already exists
        cred (str): the path of the secret key JSON file

    Example 1:
        df = pd.DataFrame({'list': [[1, 2, 3]],
                      'dict': [{'key1': 'value1'}],
                      'list_of_dict': [[{'key1': 'A', 'key2': 1}, {'key1': 'B', 'key2': 2}, {'key1': 'C', 'key2': 3}]]})

        upload_df_to_bq(df,
            'hktv-data',
            'working_data',
            'trial',
            'append')

    Example 2:
        df = pd.DataFrame({'abc': [1, 2, 3, np.NaN]})

        upload_df_to_bq(df,
            'hktv-data',
            'working_data',
            'trial',
            'replace')

    Remarks:
        pd.DataFrame.to_gbq() could handle NaN value, while pandas_gbq.to_gbq() could not
    """
    # check method input
    valid_method = {'fail', 'replace', 'append'}
    if method not in valid_method:
        raise ValueError("method must be one of {}".format(valid_method))

    # authorisation
    if cred != None:
        # Get credential from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            cred)
        logger.info("Loaded credential from the service account key")
    else:
        credentials, _ = google.auth.default()

    # replace all null as np.NaN
    df.fillna(np.NaN, inplace=True)
    df.to_gbq(".".join([dataset_id, table_id]),
              project_id=project_id,
              if_exists=method,
              credentials=credentials,
              table_schema=generate_schema(df))
    logger.info(
        "Dataframe uploaded to the destinated table: {}.{}.{} ({})".format(
            project_id, dataset_id, table_id, method))


@deprecated
def upload_df_to_bq_old(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema: dict,
    client,
):
    """
    To upload a pandas dataframe to GCP Bigquery. The upload data can contain a nested column (array/ array of struct)

    Background:
        the to_gbq function from pandas does not support nested or array values.
        updated: but the to_gbq() method of the class dataframe supports nested or array values.

    Reference:
        https://cloud.google.com/bigquery/docs/pandas-gbq-migration
        https://medium.com/inside-league/loading-complex-json-files-in-realtime-to-bigquery-from-pubsub-using-dataflow-and-updating-the-49b420838bac

    Args:
        df = the target dataframe to upload
        schema = the list of order dictionaries which describe the schema of df
        project_id = the project ID where the dataframe is going to be uploaded
        dataset_id = the dataset ID where the dataframe is going to be uploaded
        table_id = the table ID where the dataframe is going to be stored

    Example 1:
        df = pd.DataFrame({'list': [[1, 2, 3]],
                      'dict': [{'key1': 'value1'}],
                      'list_of_dict': [[{'key1': 'A', 'key2': 1}, {'key1': 'B', 'key2': 2}, {'key1': 'C', 'key2': 3}]]})

        upload_df_to_bq(df,
            'hktv-data',
            'working_data',
            'trial',
            generate_schema(df),
            bigquery.Client())

    Caution:
        columns with nan as float will raise error
    """
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)

    # df_json = df.to_dict("records")
    df_json = df.to_json(orient="records")
    # logger.info("dataframe is converted into JSON format")

    client.load_table_from_json(df_json,
                                ".".join([project_id, dataset_id, table_id]),
                                job_config=job_config).result()
    # logger.info("dataframe uploaded to {}.{}.{}".format(
    #     project_id, dataset_id, table_id))


def bq_query(project: str, query: str, cred: str = None):
    """
    Wrapper for BQ client function which you specify the query in a string format as the argument

    Args:
        query (str): SQL query
        project (str): gcp project name. e.g. hktv-data
        cred (str): the path of the secret key JSON file

    Example:
        project_id = "hktv-data"
        sql = "SELECT * FROM `hktv-data.output_intermediate.pt_consumer_council` LIMIT 10 "
        df = bq_query(project_id, sql)

    Returns:
        df: Query result in dataframe format
    """
    if cred != None:
        # Get credential from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            cred)
        logger.info("Loaded credential from the service account key")
    else:
        credentials, _ = google.auth.default()
    client = bigquery.Client(
        credentials=credentials)  # Construct a BigQuery client object.
    query_job = client.query(query=query,
                             project=project)  # Make an API request.

    return query_job.result().to_dataframe()


def bq_query_path(query_path: str,
                  keyword: str,
                  start_date: str = None,
                  end_date: str = None,
                  project_id: str = 'hktv-data',
                  cred: str = None):
    """
    Wrapper for BQ client function which you can specify the file path of the query

    Args:
        query (str): path of a SQL query
        start_date (str): start date of the targeted period in format of 'YYYYMMDD'
        end_date (str): end date of the targeted period in format of 'YYYYMMDD'
        project_id (str): gcp project name. e.g. hktv-data
        cred (str): the path of the secret key JSON file

    Example:
        project_id = "hktv-data"
        sql = "./src/sql/category_name.sql"
        df = bq_query(project_id, sql)

    Returns:
        df: Query result in dataframe format

    TODO:
        (Maybe) Use service account json instead of bigquery.Client for authentication
    """
    if cred != None:
        # Get credential from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            cred)
        logger.info("Loaded credential from the service account key")
    else:
        credentials, _ = google.auth.default()
    # Open the SQL script and load as string
    with open(query_path, 'r', encoding='utf8') as sql_file:
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
    # insert the argument "keyword" into the SQl script
    query = query.format(KEYWORD=keyword)
    # read the query from BQ
    df = pandas_gbq.read_gbq(query,
                             dialect='standard',
                             project_id=project_id,
                             credentials=credentials)
    logger.info(f"Downloaded from BQ for - {df.shape[0]} records")

    return df


def upload_file_to_gcs(bucket_name: str,
                       blob_name: str,
                       path_to_file: str,
                       project_id: str = "hktv-data",
                       cred: str = None):
    """ Upload object with a path to a GCP bucket

    Args:
        bucket_name (str): the name of the bucket on GCP
        blob_name (str): the destinated location of the file to be uploaded on GCP (need to include the folder at the path)
        path_to_file (str): the path to the file in the local
        project_id (str, optional): GCP project ID. Defaults to "hktv-data".
        cred (str, optional): the path of the secret key JSON file. Defaults to None.

    Returns:
        str: the link of the file uploaded

    Example:
        filepath = "./output/trial.xlsx"
        filename = re.search('(?<=\/)[^\/]*$', filepath).group()
        upload_file_to_gcs('hktv-simon', filename, filepath)
    """

    # Explicitly use service account credentials by specifying the private key file.
    if cred != None:
        # Get credential from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            cred)
        logger.info("Loaded credential from the service account key")
    else:
        credentials, _ = google.auth.default()
    storage_client = storage.Client(project_id, credentials=credentials)

    # print(pd.Series(list(storage_client.list_buckets())))

    # Specify the bucket and blob information
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # upload file to GCS
    blob.upload_from_filename(path_to_file)

    #returns a public url
    return blob.public_url


def upload_df_to_gcs(
    df: pd.DataFrame,
    bucket_name: str,
    blob_name: str,
    project_id: str = "hktv-data",
    cred: str = None,
):
    """To upload a pandas dataframe to GCS Storage in CSV format. This can avoid to save the dataframe at local.

    Args:
        df (pd.DataFrame): the target dataframe to upload
        blob_name (str): the destinated location of the file to be uploaded on GCP (need to include the folder at the path)
        path_to_file (str): the path to the file in the local
        project_id (str, optional): GCP project ID. Defaults to "hktv-data".
        cred (str, optional): the path of the secret key JSON file. Defaults to None.

    Returns:
        str: the link of the file uploaded

    Example:
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ["A", "B", "C"]})
        upload_df_to_gcs(df, bucket_name='hktv-simon', blob_name="upload_df_to_gcs.csv")
    """
    # Explicitly use service account credentials by specifying the private key file.
    if cred != None:
        # Get credential from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            cred)
        logger.info("Loaded credential from the service account key")
    else:
        credentials, _ = google.auth.default()
    storage_client = storage.Client(project_id, credentials=credentials)

    # Specify the bucket and blob information
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # upload dataframe to GCS as a csv
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')

    #returns a public url
    return blob.public_url


if __name__ == "__main__":
    read_gs()
    upload_df_to_bq()
    upload_file_to_gcs()
    upload_df_to_gcs()
    bq_query()
    bq_query_path()