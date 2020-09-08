"""
Helper Functions to create and manage SQLITE Database for Covid19 Project.
data_to_df(table_name, fetch_new = False) to fetch data from a table in the form of a Pandas DF.

Run Directly to ingest data to default SQLite DB. Import and Use data_to_df to grab data from DB as a Pandas DF.

Dependencies : india_API_data/Covid19_india_org_api.py for functions that fetch data from API.

References :
https://stackoverflow.com/questions/14431646/how-to-write-pandas-dataframe-to-sqlite-with-index
https://stackoverflow.com/questions/39407254/how-to-set-the-primary-key-when-writing-a-pandas-dataframe-to-a-sqlite-database
"""

from india_API_data.Covid19_india_org_api import get_test_dataframe, make_dataframe
import pandas as pd
import sqlite3
from sqlite3 import OperationalError


def fetch_data():
    """Fetches the Latest Stats and Testing Data and returns a clean DF
    ready to be sent to the DB.
    """
    # Fetching the Updated Data
    cases = make_dataframe()
    tests = get_test_dataframe()

    india_combined_samples = tests.join(cases, how='right')
    india_combined_samples.reset_index(inplace=True)
    india_combined_samples.rename({'index': 'date'}, axis=1, inplace=True)
    return india_combined_samples


def create_table(cursor, table_name):
    """ Requires cursor to be passed. Creates a table with the
    necessary columns for the covid data if it doesn't exist.
    """
    try:
        cursor.execute(f""" CREATE TABLE {table_name} (
        date DateTime PRIMARY KEY,
        TestingSamples integer,
        DailyConfirmed integer,
        DailyDeceased integer, 
        DailyRecovered integer,
        TotalConfirmed integer,
        TotalDeceased integer,
        TotalRecovered integer)""")

        cursor.execute(f"SELECT name FROM PRAGMA_TABLE_INFO({table_name})")
        print(f'Created {table_name} with columns : {cursor.fetchall()}')

    except OperationalError:
        cursor.execute(f"""SELECT * FROM {table_name}""")
        num_records = len(cursor.fetchall())
        print(f'Table Exists With {num_records} Records')


def add_data_table(conn, tablename, df):
    """Appends New Covid Data to table if it exists
    else Creates a new table and appends all data.
    Takes in connection to DB.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT * FROM {tablename}""")

    except OperationalError:
        create_table(cursor, tablename)
        cursor.execute(f"""SELECT * FROM {tablename}""")
    finally:
        num_records = len(cursor.fetchall())
        df.iloc[num_records:].to_sql(tablename, conn, if_exists='append', index=False)
        print(f'Added {len(df) - num_records} Records')


def data_to_df(table_name, fetch_new=False):
    """ Returns the complete covid data from the table as a dataframe.
    If fetch_new = True, First fetches the data from API, updates Database and then returns DF.
    Default table name 'covid'.
    If table doesn't exist, creates one. Must be used with fetch_new = True to get data into new table.
    """
    # Create/load DB
    database = 'Data/covid_data.db'
    # Open connection to DB
    conn = sqlite3.connect(database)
    # create client/cursor
    client = conn.cursor()

    # if new data requested
    if fetch_new:
        # Fetch the data
        india_data = fetch_data()
        # Create Table Named Covid
        create_table(client, table_name)
        # Add data/Append New Data to table
        add_data_table(conn, table_name, india_data)

    # fetch data from table
    return pd.read_sql_query(f"SELECT * from {table_name}", conn)
    # Close connection
    conn.close()


if __name__ == '__main__':
    # Create/load DB
    database = 'Data/covid_data.db'

    # Open connection to DB
    conn = sqlite3.connect(database)
    # Create Client or Cursor
    client = conn.cursor()
    # Fetch the data
    india_data = fetch_data()

    # Create Table Named Covid
    create_table(client, 'covid')

    # Add data/Append New Data to table
    add_data_table(conn, 'covid', india_data)

    # Close connection
    conn.close()
