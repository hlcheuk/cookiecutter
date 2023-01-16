from typing import Any, List, Union
import errno
import yaml, toml, os, sys
from pathlib import Path
import pandas as pd
import pandas_gbq
import regex as re
# import warnings
from google.cloud import bigquery
import functools
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib, ssl

from src.utils.logger import setup_logger

__all__ = "Utils"
logger = setup_logger(__all__)


def list_full_paths(directory: str):
    """To list the absolute paths of the specified directory

    Args:
        directory (str): the path of the folder

    Returns:
        list: list of the absolute path of the files
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)]


def get_config(filename: str, key: str):
    """
    Get from toml file, return values with the key provided

    Args:
        filename (str): Name of the config file. used to look up config file (e.g. ./config/{filename}.toml
        key (str): Key to be looked up in the config file (e.g. threshold)

    Returns:
        res (any): values from the config file with the provided key (e.g. [0.8, 1.0, 1.1])

    Example:
        res = get_config("setting", "threshold")

    Raise:
        FileNotExist
        KeyError
    """
    path = os.path.join("config", "{}.toml".format(filename.lower()))

    if os.path.isfile(path) is False:  # If file does not exist
        raise FileNotFoundError(f"File not found, please check. {path}")

    config_dic = toml.load(path)
    config_value = config_dic.get(key)

    if config_value is None:  # If config key exists
        raise KeyError(f"Key - {key} not found in {path}. Please check")

    return config_value


def check_directory(path: str, isdir: bool = False):
    """
    Check if the base directory exists. If not, create it recursively

    Args:
        path (str): Full path of local file. (e.g. /Git/pde/data/abc.csv)
        isdir (bool): A bool to indicate whether the path is a directory

    Returns:
        res (bool): True

    TODO:
        - raise error if cannot create folder
        - Check for directory case, whether the path already has a trailing slash
    """

    if isdir is True:
        path = path + "/"

    directory = os.path.dirname(path)

    if os.path.exists(directory):
        return True

    else:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)

        except OSError as e:
            msg = "Fatal: output directory does not exist and cannot be created - {}".format(
                directory)
            logger.error(e)
            logger.error(msg)
            exit(1)

        logger.info("Created local directory - {}".format(path))

    return True


def get_env(key: str):
    """
    Get environment variable if exists

    Args:
        NA

    Returns:
        res (bool): True

    Example:
        get_env("GOOGLE_APPLICATION_CREDENTIALS")

    Error:
        KeyError: Environment variable not set
        FileNotFoundError:

    TODO:
        Handle file not found
    """
    try:
        credential_file = os.environ[key]

    except KeyError:
        logger.error(f"Please set the environment variable {key}")
        sys.exit(1)

    if os.path.exists(credential_file):
        pass
    else:
        logger.error(
            f"Cloud credential file does not exists. Please check - {credential_file} \n\n"
        )

        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                credential_file)

    return os.environ[key]


def export_feather(df, filename):
    """
    A wrapper to export dataframe to output folder

    Args:
        df (Dataframe): Pandas Dataframe
        filename (str): Name of the feather file. e.g. sales.feather
    """
    path = f"output/{filename}"
    check_directory(path)  # Check if the directory exists
    df.to_feather(path)
    logger.info(f"Saved output - {path} - {df.shape[0]} records")


def write_sql_to_bq(sql: str, table: str, partition_col: str):
    """
    Write sql result directly to the new bigquery table (Create only. Need manually delete the table if exists)

    Args:
        sql (str): sql statements
        table (str): Destination table. e.g. hktv-data.output_intermediate.test
        partition_col (str): Partition column. e.g. date

    Return:
        None. Result should be inserted to the bq table

    Example:
        sql = '''
        SELECT [a, b, c] a, date
        FROM
        (SELECT 5 AS a,
                37 AS b,
                406 AS c,
                CURRENT_DATE() AS date);
        '''
        table = "hktv-data.output_intermediate.test"
        write_sql_to_bq(sql=sql, table=table, partition_col="date")

    Exceptions:
        - Table already exists
              google.api_core.exceptions.Conflict: 409 Already Exists

        - Partition column not exists
              google.api_core.exceptions.BadRequest: 400 The field specified for partitioning cannot be found in the schema

    """

    client = bigquery.Client()
    table_id = table

    # Set column for partitioning. Usually the date column
    if partition_col is not None:
        partition = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_col,
            expiration_ms=None,
            require_partition_filter=None,
        )
        job_config = bigquery.QueryJobConfig(destination=table_id,
                                             time_partitioning=partition)
    else:
        job_config = bigquery.QueryJobConfig(destination=table_id)

    # Send job with config
    query_job = client.query(
        sql,
        job_config=job_config,
    )
    query_job.result()

    print("Query results loaded to the table {}".format(table_id))
    return None


def upload_to_bq(df: pd.DataFrame,
                 project_id: str,
                 table: str,
                 if_exists: str = "append"):
    """
    Upload pandas dataframe to bigquery table

    Args:
        df (Dataframe): Pandas Dataframe
        project_id (str): Name of gcp project. e.g. hktv-data
        table (str): Name of the destination table. e.g. output_intermediate.cross_cat
        if_exists (str): Action when table already exists. e.g. fail/replace/append

    Returns:
        None

    Reference:
        Date column schema
        https://stackoverflow.com/questions/52784157/how-to-convert-a-panda-column-into-big-query-table-date-format

    Example:
        utils.upload_to_bq(
            df=test,
            project_id="hktv-data",
            table="output_intermediate.cross_cat",
            if_exists="replace",
        )
    """

    logger.info(f"Inserting to bigquery - [{project_id} - {table}]")
    pandas_gbq.to_gbq(df,
                      destination_table=table,
                      project_id=project_id,
                      if_exists=if_exists)
    logger.info(f"Finished Insert - {df.shape[0]} records")


def yaml2json(yaml_path: str):
    with open(yaml_path, "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)


def send_email(sender: str, email: str, attachments: list):
    """Send email via SMTP

    Args:
        sender (str): path to the config file (YAML) of the sender login information
        email (str): path to the config file (YAML) of the email information
        attachments (list): list of paths of attachments

    Example:
        send_email("config/sender.yaml", "config/email.yaml", ["output/warehouse_demand_forecast_20221101.xlsx"])
    """
    # read sender login file
    config = yaml2json(sender)
    login = {}
    login["user"] = config["USERNAME"]
    login["password"] = config["PASSWORD"]
    sender_name = config["SENDER_NAME"]
    hostname = config["HOSTNAME"]
    # get email information
    email = yaml2json(email)
    recipients = email["recipients"]
    subject = email["subject"]
    # construct email
    mail = MIMEMultipart()
    mail['From'] = sender_name
    mail['To'] = (', ').join(recipients)
    mail['Subject'] = subject
    # write text
    mail.attach(MIMEText('FYI.', 'plain', 'UTF-8'))
    # attach report
    for a in attachments:
        filename = re.search("[^\/]*$", a).group()
        attachment = MIMEApplication(open(a, 'rb').read())
        attachment.add_header('Content-Disposition',
                              'attachment',
                              filename=filename)
        mail.attach(attachment)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(hostname, port=465, context=ctx) as server:
        server.login(**login)
        _mail_info = {
            'To': mail['To'],
            'recipients': recipients,
            'subject': mail['Subject']
        }
        server.sendmail(sender_name, recipients, mail.as_string())
        logger.info(f'Sent email: {_mail_info}')


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        logger.warn("Call to deprecated function {}.".format(func.__name__))
        return func(*args, **kwargs)

    return new_func


if __name__ == "__main__":
    list_full_paths()
    get_config()
    check_directory()
    get_env()
    export_feather()
    write_sql_to_bq()
    upload_to_bq()
    send_email()