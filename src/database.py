import os
import pandas as pd
import urllib
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():

    params = urllib.parse.quote_plus(
        f"DRIVER=ODBC Driver 13 for SQL Server;"
        f"SERVER={os.getenv('DB_HOST')},{os.getenv('DB_PORT')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        f"UID={os.getenv('DB_USER')};"
        f"PWD={os.getenv('DB_PASS')};"
    )

    connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

    engine = create_engine(connection_string)

    return engine


def load_training_data():

    engine = get_engine()

    query = """
        SELECT *
        FROM dbo.EXTRACAO_DADOS_SISTEMA
    """

    df = pd.read_sql(query, engine)

    return df