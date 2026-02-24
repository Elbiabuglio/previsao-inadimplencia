import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_connection():

    conn = pyodbc.connect(
        f"DRIVER={{ODBC Driver 13 for SQL Server}};"
        f"SERVER={os.getenv('DB_HOST')},{os.getenv('DB_PORT')};"
        f"DATABASE={os.getenv('DB_NAME')};"
        f"UID={os.getenv('DB_USER')};"
        f"PWD={os.getenv('DB_PASS')};"
    )

    return conn


def load_training_data():

    conn = get_connection()

    query = """
        SELECT *
        FROM dbo.EXTRACAO_DADOS_SISTEMA
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df