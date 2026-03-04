import os
import logging
import urllib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

# ==========================================
# CONFIGURAÇÃO DE LOG
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# VALIDAÇÃO DE VARIÁVEIS DE AMBIENTE
# ==========================================
def _validate_env_variables():
    required_vars = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS"]
    
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Variável de ambiente {var} não definida.")


# ==========================================
# CRIAR ENGINE
# ==========================================
def get_engine():
    """
    Cria e retorna uma engine de conexão com SQL Server.
    """
    try:
        _validate_env_variables()

        logger.info("Criando engine de conexão com o banco...")

        params = urllib.parse.quote_plus(
            f"DRIVER=ODBC Driver 13 for SQL Server;"
            f"SERVER={os.getenv('DB_HOST')},{os.getenv('DB_PORT')};"
            f"DATABASE={os.getenv('DB_NAME')};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASS')};"
            f"TrustServerCertificate=yes;"
        )

        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"

        engine = create_engine(
            connection_string,
            pool_pre_ping=True,  # evita conexões quebradas
            pool_recycle=3600
        )

        logger.info("Engine criada com sucesso.")
        return engine

    except Exception as e:
        logger.error(f"Erro ao criar engine: {e}")
        raise


# ==========================================
# TESTAR CONEXÃO
# ==========================================
def test_connection():
    """
    Testa a conexão com o banco.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Conexão com banco testada com sucesso.")
        return True
    except Exception as e:
        logger.error(f"Falha ao testar conexão: {e}")
        return False


# ==========================================
# CARREGAR DADOS PARA TREINO
# ==========================================
def load_training_data():
    """
    Carrega os dados da tabela dbo.EXTRACAO_DADOS_SISTEMA.
    """
    try:
        logger.info("Iniciando carregamento dos dados...")

        query = """
            SELECT *
            FROM dbo.EXTRACAO_DADOS_SISTEMA
        """

        engine = get_engine()

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")

        return df

    except SQLAlchemyError as e:
        logger.error(f"Erro SQLAlchemy ao carregar dados: {e}")
        raise

    except Exception as e:
        logger.error(f"Erro inesperado ao carregar dados: {e}")
        raise


# ==========================================
# INSERIR DATAFRAME EM TABELA
# ==========================================
def insert_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "append"):
    """
    Insere um DataFrame em uma tabela do banco.
    """
    try:
        engine = get_engine()
        logger.info(f"Inserindo {len(df)} registros na tabela {table_name}...")

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=500
        )

        logger.info("Inserção realizada com sucesso.")

    except Exception as e:
        logger.error(f"Erro ao inserir dados: {e}")
        raise


# ==========================================
# LIMPAR TABELA
# ==========================================
def truncate_table(table_name: str):
    """
    Executa TRUNCATE TABLE na tabela informada.
    """
    try:
        engine = get_engine()

        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name}"))

        logger.info(f"Tabela {table_name} limpa com sucesso.")

    except Exception as e:
        logger.error(f"Erro ao truncar tabela {table_name}: {e}")
        raise


# ==========================================
# SALVAR PREVISÕES EM PRODUÇÃO
# ==========================================
def save_predictions(df: pd.DataFrame, table_name: str = "RESULTADOS_INTERMEDIARIOS"):
    """
    Salva as previsões no banco:
    1️⃣ Limpa a tabela
    2️⃣ Insere o DataFrame
    """
    try:
        logger.info(f"Iniciando salvamento de previsões na tabela {table_name}...")


        # Insere dados
        insert_dataframe(df, table_name, if_exists="append")

        logger.info(f"Previsões salvas com sucesso na tabela {table_name}.")

    except Exception as e:
        logger.error(f"Erro ao salvar previsões: {e}")
        raise


# ==========================================
# EXECUTAR STORED PROCEDURE
# ==========================================
def execute_procedure(procedure_name: str):
    """
    Executa uma stored procedure no banco.
    """
    try:
        engine = get_engine()

        with engine.begin() as conn:
            conn.execute(text(f"EXEC {procedure_name}"))

        logger.info(f"Procedure {procedure_name} executada com sucesso.")

    except Exception as e:
        logger.error(f"Erro ao executar procedure {procedure_name}: {e}")
        raise