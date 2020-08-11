from google.cloud import storage
import sqlalchemy
import pandas as pd
import pg8000  # databse driver
import os


def download_folder_bucket(bucket, bucket_folder, local_folder):
    """Download all files from a GCS bucket folder to a local folder.
    """
    # list of filenames in bucket_folder
    file_list = [file.name for file in bucket.list_blobs(prefix=bucket_folder)]

    # iterate over blobs and doenload to local folder + filename

    for file in file_list:
        blob = bucket.blob(file)
        # filename by splitting name by '/' and keeping last item
        filename = blob.name.split('/')[-1]
        # download to local folder
        blob.download_to_filename(local_folder + filename)
    return f'Downloaded {len(file_list)} Files'


def add_data_table(engine, tablename, df):
    """ Appends New Data to table if it exists
    Takes in engine connected to DB, tablename and dataframe to store.
    Throws error if 1. Table Doesn't Exist, 2. incorrect table and dataframe ?(abstract this coice away from user)
    Problematic for testing_table as it has duplicates. - Possible solution, find last index and not length.
    """

    try:
        results = engine.execute(f"""SELECT * FROM {tablename}""")
        num_records = len(results.fetchall())
        print(f'{num_records} Records in {tablename}')

        df[num_records:].to_sql(tablename, engine, if_exists='append')
        print(f'Added {len(df[num_records:])} Records to table')

    # Just can't seem to get errors to work
    except:
        print('Errored. Investigate')


# Create SQLAlchemy connection to CloudSQL Server.

# Remember - storing secrets in plaintext is potentially unsafe.

def connect_db():
    """ Connects to Cloud SQL DB Using provided Unix Socket. Username, Password etc. Hardcoded.
    Problematic.
    """
    db_user = 'postgres'
    db_pass = '1urAug2szewJrvng'
    db_name = 'covid19-data'
    db_socket_dir = os.environ.get("DB_SOCKET_DIR", "/cloudsql")
    cloud_sql_connection_name = 'covid19-india-analysis-284814:asia-south1:covid19-data-server'

    engine = sqlalchemy.create_engine(
        # Equivalent URL:
        # postgres+pg8000://<db_user>:<db_pass>@/<db_name>
        #                         ?unix_sock=<socket_path>/<cloud_sql_instance_name>/.s.PGSQL.5432
        sqlalchemy.engine.url.URL(
            drivername="postgres+pg8000",
            username=db_user,  # e.g. "my-database-user"
            password=db_pass,  # e.g. "my-database-password"
            database=db_name,  # e.g. "my-database-name"
            query={
                "unix_sock": "{}/{}/.s.PGSQL.5432".format(
                    db_socket_dir,  # e.g. "/cloudsql"
                    cloud_sql_connection_name)  # i.e "<PROJECT-NAME>:<INSTANCE-REGION>:<INSTANCE-NAME>"
            }
        ),
        # ... Specify additional properties here.
    )

    return engine


def main(request):
    """ Driver function for CLoud Function. Request doesn't do anything.
    """
    # Create GCS client
    storage_client = storage.Client()

    # connect to a bucket
    bucket = storage_client.get_bucket('covid19-india-analysis-bucket')

    # Download RAW CSVs from GCS Bucket to Cloud Function temp. storage.
    download_folder_bucket(bucket, 'Data/Raw/', '/tmp/')

    # Loading and Transforming data
    data = pd.read_csv('/tmp/COVID_India_National.csv', parse_dates=True, index_col=0)
    state = pd.read_csv('/tmp/COVID_India_State.csv', parse_dates=True, index_col=0)
    # Load and clean test data
    test = pd.read_csv('/tmp/COVID_India_Test_data.csv', parse_dates=True, index_col=0)
    test = test.loc[~test.index.duplicated(keep='last')]

    # Connect to CloudSQL DB
    engine = connect_db()

    # Uploading Data to DB
    add_data_table(engine, 'overall_stats', data)
    add_data_table(engine, 'states_info', state)
    add_data_table(engine, 'testing_stats', test)

    print('Executed')