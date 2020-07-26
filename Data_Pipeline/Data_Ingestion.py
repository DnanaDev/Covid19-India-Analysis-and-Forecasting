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
    "Date" DATE PRIMARY KEY,
    "Total.Confirmed" INTEGER,
      "Total.Deceased" INTEGER,
      "Total.Recovered" INTEGER,
      "AndamanAndNicobarIslands.Confirmed" INTEGER,
      "AndamanAndNicobarIslands.Deceased" INTEGER,
      "AndamanAndNicobarIslands.Recovered" INTEGER,
      "AndhraPradesh.Confirmed" INTEGER,
      "AndhraPradesh.Deceased" INTEGER,
      "AndhraPradesh.Recovered" INTEGER,
      "ArunachalPradesh.Confirmed" INTEGER,
      "ArunachalPradesh.Deceased" INTEGER,
      "ArunachalPradesh.Recovered" INTEGER,
      "Assam.Confirmed" INTEGER,
      "Assam.Deceased" INTEGER,
      "Assam.Recovered" INTEGER,
      "Bihar.Confirmed" INTEGER,
      "Bihar.Deceased" INTEGER,
      "Bihar.Recovered" INTEGER,
      "Chandigarh.Confirmed" INTEGER,
      "Chandigarh.Deceased" INTEGER,
      "Chandigarh.Recovered" INTEGER,
      "Chhattisgarh.Confirmed" INTEGER,
      "Chhattisgarh.Deceased" INTEGER,
      "Chhattisgarh.Recovered" INTEGER,
      "DadraAndNagarHaveliAndDamanAndDiu.Confirmed" INTEGER,
      "DadraAndNagarHaveliAndDamanAndDiu.Deceased" INTEGER,
      "DadraAndNagarHaveliAndDamanAndDiu.Recovered" INTEGER,
      "Dd.Confirmed" INTEGER,
      "Dd.Deceased" INTEGER,
      "Dd.Recovered" INTEGER,
      "Delhi.Confirmed" INTEGER,
      "Delhi.Deceased" INTEGER,
      "Delhi.Recovered" INTEGER,
      "Goa.Confirmed" INTEGER,
      "Goa.Deceased" INTEGER,
      "Goa.Recovered" INTEGER,
      "Gujarat.Confirmed" INTEGER,
      "Gujarat.Deceased" INTEGER,
      "Gujarat.Recovered" INTEGER,
      "Haryana.Confirmed" INTEGER,
      "Haryana.Deceased" INTEGER,
      "Haryana.Recovered" INTEGER,
      "HimachalPradesh.Confirmed" INTEGER,
      "HimachalPradesh.Deceased" INTEGER,
      "HimachalPradesh.Recovered" INTEGER,
      "JammuAndKashmir.Confirmed" INTEGER,
      "JammuAndKashmir.Deceased" INTEGER,
      "JammuAndKashmir.Recovered" INTEGER,
      "Jharkhand.Confirmed" INTEGER,
      "Jharkhand.Deceased" INTEGER,
      "Jharkhand.Recovered" INTEGER,
      "Karnataka.Confirmed" INTEGER,
      "Karnataka.Deceased" INTEGER,
      "Karnataka.Recovered" INTEGER,
      "Kerala.Confirmed" INTEGER,
      "Kerala.Deceased" INTEGER,
      "Kerala.Recovered" INTEGER,
      "Ladakh.Confirmed" INTEGER,
      "Ladakh.Deceased" INTEGER,
      "Ladakh.Recovered" INTEGER,
      "Lakshadweep.Confirmed" INTEGER,
      "Lakshadweep.Deceased" INTEGER,
      "Lakshadweep.Recovered" INTEGER,
      "MadhyaPradesh.Confirmed" INTEGER,
      "MadhyaPradesh.Deceased" INTEGER,
      "MadhyaPradesh.Recovered" INTEGER,
      "Maharashtra.Confirmed" INTEGER,
      "Maharashtra.Deceased" INTEGER,
      "Maharashtra.Recovered" INTEGER,
      "Manipur.Confirmed" INTEGER,
      "Manipur.Deceased" INTEGER,
      "Manipur.Recovered" INTEGER,
      "Meghalaya.Confirmed" INTEGER,
      "Meghalaya.Deceased" INTEGER,
      "Meghalaya.Recovered" INTEGER,
      "Mizoram.Confirmed" INTEGER,
      "Mizoram.Deceased" INTEGER,
      "Mizoram.Recovered" INTEGER,
      "Nagaland.Confirmed" INTEGER,
      "Nagaland.Deceased" INTEGER,
      "Nagaland.Recovered" INTEGER,
      "Odisha.Confirmed" INTEGER,
      "Odisha.Deceased" INTEGER,
      "Odisha.Recovered" INTEGER,
      "Puducherry.Confirmed" INTEGER,
      "Puducherry.Deceased" INTEGER,
      "Puducherry.Recovered" INTEGER,
      "Punjab.Confirmed" INTEGER,
      "Punjab.Deceased" INTEGER,
      "Punjab.Recovered" INTEGER,
      "Rajasthan.Confirmed" INTEGER,
      "Rajasthan.Deceased" INTEGER,
      "Rajasthan.Recovered" INTEGER,
      "Sikkim.Confirmed" INTEGER,
      "Sikkim.Deceased" INTEGER,
      "Sikkim.Recovered" INTEGER,
      "TamilNadu.Confirmed" INTEGER,
      "TamilNadu.Deceased" INTEGER,
      "TamilNadu.Recovered" INTEGER,
      "Telangana.Confirmed" INTEGER,
      "Telangana.Deceased" INTEGER,
      "Telangana.Recovered" INTEGER,
      "Tripura.Confirmed" INTEGER,
      "Tripura.Deceased" INTEGER,
      "Tripura.Recovered" INTEGER,
      "UttarPradesh.Confirmed" INTEGER,
      "UttarPradesh.Deceased" INTEGER,
      "UttarPradesh.Recovered" INTEGER,
      "Uttarakhand.Confirmed" INTEGER,
      "Uttarakhand.Deceased" INTEGER,
      "Uttarakhand.Recovered" INTEGER,
      "WestBengal.Confirmed" INTEGER,
      "WestBengal.Deceased" INTEGER,
      "WestBengal.Recovered" INTEGER,
      "StateUnassigned.Confirmed" INTEGER,
      "StateUnassigned.Deceased" INTEGER,
      "StateUnassigned.Recovered" INTEGER,
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
    engine = create_engine('postgresql://postgres:Anand1996@localhost:5432/Covid19-India')

    # Creating Tables (if run for the first time)
    # create_table_overall_stats(engine)
    # create_table_testing_stats(engine)
    # create_table_state_info(engine)

    # Ingesting RAW data and transforming dataframes for Database Ingestion
    # 1. National data
    data = make_dataframe(save=True)

    # 2. Testing Data - has duplicates for a single date, will fail the unique constraint for key, removing.
    test = get_test_dataframe(save=True)
    test = test.loc[~test.index.duplicated(keep='last')]

    # 3. States data - Modifying/flattening multi-index for state, stat
    state = make_state_dataframe(save=True)
    cols = state.columns.get_level_values(0).str.title() + '.' + state.columns.get_level_values(1)
    state.columns = cols
    state.columns = state.columns.str.replace(' ', '')

    # Adding data to tables

    add_data_table(engine, 'overall_stats', data)
    add_data_table(engine, 'testing_stats', test[:-1])  # remove -1 if update after midnight, due to mismatched
    # frequency of update
    add_data_table(engine, 'states_info', state)

    # local backup of DB - Currently overwrites previous backup, do acc. to time/data
    backup_db('/Users/apple/Desktop/DS/Covid19-Kaggle_and_End-End_project/Data/Cleaned/Covid19-India_backup.sql')
