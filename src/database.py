import os
import logging
import urllib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


def get_engine():
    """
    Cria e retorna uma engine de conexão com o SQL Server
    utilizando as variáveis definidas no arquivo .env.
    """

    try:
        logger.info("Criando engine de conexão com o banco...")

        params = urllib.parse.quote_plus(
            f"DRIVER=ODBC Driver 13 for SQL Server;"
            f"SERVER={os.getenv('DB_HOST')},{os.getenv('DB_PORT')};"
            f"DATABASE={os.getenv('DB_NAME')};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASS')};"
        )

        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

        engine = create_engine(connection_string)

        logger.info("Engine criada com sucesso.")

        return engine

    except Exception as e:
        logger.error(f"Erro ao criar engine: {e}")
        raise


def load_training_data():
    """
    Carrega os dados da tabela dbo.EXTRACAO_DADOS_SISTEMA
    e retorna um DataFrame pandas.
    """

    try:
        logger.info("Iniciando carregamento dos dados...")

        engine = get_engine()

        query = """
            SELECT *
            FROM dbo.EXTRACAO_DADOS_SISTEMA
        """

        df = pd.read_sql(query, engine)

        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")

        return df

    except SQLAlchemyError as e:
        logger.error(f"Erro de SQLAlchemy ao carregar dados: {e}")
        raise

    except Exception as e:
        logger.error(f"Erro inesperado ao carregar dados: {e}")
        raise