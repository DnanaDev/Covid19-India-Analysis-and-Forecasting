""" Data Ingestion for Covid19 Data Pipeline

Using SQLAlchemy engine to interface to PostgresQL Database.
Functions to create DB according to schema and for ingesting data.
The use case is to run the script and automatically to update CSVs in Data/Raw and to
store the cleaned data in the database. Backup of the database in stored in Data/cleaned.

# Data Ingestion Functions
1. add_data_table(engine, tablename, df)
Uses Pandas dataframe from Covid19_india_org_api to append data to table using SQLAlchemy and DF.to_sql()
Issues :
1. append problem, duplicate key values. Shouldn't happen with append parameter in to_sql(), but here we are.
workaround - fetch length of existing records in table and then only store records after that. Can be problematic.
2. Cannot replace due to the presence of foreign key.

Dependencies : Uses data fetching functions from script - Covid19_india_org_api

To Do/ Extensions :
1. Data Validation not done anywhere.
2. No error handling done for SQLAlchemy functions. (key constraint etc.)
2. Modify for Google Cloud SQL
3. Connecting to Google Cloud Functions.
"""

from sqlalchemy import create_engine
import pandas as pd
from Covid19_india_org_api import make_dataframe, get_test_dataframe, make_state_dataframe
from psycopg2 import ProgrammingError, errors, IntegrityError
import subprocess


# Creating Tables

def create_table_overall_stats(engine):
    """ Initial setup of overall_stats table according to Schema
    (rigid, hard-coded, can cause problems) - consult others. 
    """
    # Creating Overall_stats table
    engine.execute(""" CREATE TABLE overall_stats(
                "Date" DATE PRIMARY KEY,
                "DailyConfirmed" INT NOT NULL,
                "DailyDeceased" INT NOT NULL,
                "DailyRecovered" INT NOT NULL,
                "TotalConfirmed" INT NOT NULL,
                "TotalDeceased" INT NOT NULL,
                "TotalRecovered" INT NOT NULL
                )""")


def create_table_testing_stats(engine):
    """ Initial setup of testing_stats table.
    """
    # Creating testing stats table
    engine.execute(""" CREATE TABLE testing_stats(
                "Date" DATE PRIMARY KEY,
                "TestingSamples" INT NOT NULL,
                FOREIGN KEY("Date")
                    REFERENCES overall_stats("Date")
                )""")


def create_table_state_info(engine):
    """ Initial setup of state_info table, used pandas.io.sql.get_schema to create schema and added
    keys later due to the number of columns. 
    """
    # Creating state_info table
    engine.execute("""CREATE TABLE "states_info" (
    "Date" DATE ,
    "State" TEXT,
    "Confirmed" INTEGER,
    "Deceased" INTEGER,
    "Recovered" INTEGER,
    PRIMARY KEY("Date", "State"),
    FOREIGN KEY("Date")
    REFERENCES overall_stats("Date")
    )
    """)


# Data Ingestion Functions

def add_data_table(engine, tablename, df):
    """ Appends New Data to table if it exists
    Takes in engine connected to DB, tablename and dataframe to store.
    To Do :
    Throws error if 1. Table Doesn't Exist, 2. incorrect table and dataframe ?(abstract this choice away from user)
    """

    try:
        results = engine.execute(f"""SELECT * FROM {tablename}""")
        num_records = len(results.fetchall())
        print(f'{num_records} Records in {tablename}')

        df[num_records:].to_sql(tablename, engine, if_exists='append')
        print(f'Added {len(df[num_records:])} Records to table')

    # Just can't seem to get errors to work 
    except IntegrityError as e:
        print(e)
        if err == IntegrityError:
            print('Update Master Table first')


# Database local backup

def backup_db(path):
    subprocess.run(['pg_dump', '--host=localhost', '--dbname=Covid19-India',
                    '--username=postgres', '--no-password', '--format=p',
                    f'--file={path}'])
    print('Database backup complete')


# Main Function - Connect to local DB and update/ingest data if run directly.

if __name__ == '__main__':
    # Create Engine and connect to DB
    engine = create_engine('postgresql://postgres:<Pass>@localhost:5432/Covid19-India')

    # Creating Tables (if run for the first time)
    # create_table_overall_stats(engine)
    # create_table_testing_stats(engine)
    # create_table_state_info(engine)

    # Ingesting RAW data and transforming dataframes for Database Ingestion
    # 1. National data
    data = make_dataframe()

    # 2. Testing Data - has duplicates for a single date, will fail the unique constraint for key, removing.
    test = get_test_dataframe()
    test = test.loc[~test.index.duplicated(keep='last')]

    # 3. States data
    state = make_state_dataframe()

    # Adding data to tables

    add_data_table(engine, 'overall_stats', data)
    add_data_table(engine, 'testing_stats', test)  # remove -1 if update after midnight, due to mismatched
    # frequency of update
    add_data_table(engine, 'states_info', state)

    # local backup of DB - Currently overwrites previous backup, do acc. to time/data
    backup_db('/Users/apple/Desktop/DS/mentorskool/covid-19-DnanaDev/Data/Cleaned/Covid19-India_backup.sql')
