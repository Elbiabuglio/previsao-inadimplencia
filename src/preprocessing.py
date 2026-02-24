import logging
import pandas as pd


logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame):
    """
    Realiza o pré-processamento básico dos dados:
    - Converte variável alvo para binário
    - Remove colunas identificadoras
    - Separa X e y
    """

    try:
        logger.info("Iniciando pré-processamento...")

        df = df.copy()

        # ======================
        # Tratamento da variável alvo
        # ======================

        if "INADIMPLENTE_COBRANCA" not in df.columns:
            raise ValueError("Coluna INADIMPLENTE_COBRANCA não encontrada.")

        df["INADIMPLENTE_COBRANCA"] = (
            df["INADIMPLENTE_COBRANCA"]
            .str.upper()
            .map({"SIM": 1, "NAO": 0})
        )

        # ======================
        # Remover identificadores
        # ======================

        if "NUMERO_CONTRATO" in df.columns:
            df = df.drop(columns=["NUMERO_CONTRATO"])

        # ======================
        # Separar X e y
        # ======================

        X = df.drop(columns=["INADIMPLENTE_COBRANCA"])
        y = df["INADIMPLENTE_COBRANCA"]

        logger.info(f"Pré-processamento finalizado. X: {X.shape}, y: {y.shape}")

        return X, y

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        raise