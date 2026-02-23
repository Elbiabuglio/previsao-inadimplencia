import os
from dotenv import load_dotenv
import pyodbc
import pandas as pd

load_dotenv()  # carrega variáveis do .env

conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={os.getenv('DB_HOST')},{os.getenv('DB_PORT')};"
    f"DATABASE={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASS')}"
)

conn = pyodbc.connect(conn_str)
df = pd.read_sql("SELECT TOP 10 * FROM EXTRACAO_DADOS_SISTEMA", conn)
print(df.head())