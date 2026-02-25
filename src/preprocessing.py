import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame):

    try:
        logger.info("Iniciando pré-processamento...")
        df = df.copy()

        # ======================
        # 1. Tratamento da variável alvo
        # ======================

        if "INADIMPLENTE_COBRANCA" not in df.columns:
            raise ValueError("Coluna INADIMPLENTE_COBRANCA não encontrada.")

        df["INADIMPLENTE_COBRANCA"] = (
            df["INADIMPLENTE_COBRANCA"]
            .astype(str)
            .str.upper()
            .map({"SIM": 1, "NAO": 0})
        )

        # 🔥 Separar y antes de qualquer transformação
        y = df["INADIMPLENTE_COBRANCA"]
        df.drop(columns=["INADIMPLENTE_COBRANCA"], inplace=True)

        # ======================
        # 2. Remover identificadores
        # ======================

        if "NUMERO_CONTRATO" in df.columns:
            df.drop(columns=["NUMERO_CONTRATO"], inplace=True)

        # ======================
        # 3. Tratamento de datas (APENAS colunas datetime)
        # ======================

        hoje = datetime.today()

        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns

        for col in datetime_cols:
            df[f"{col}_dias"] = (hoje - df[col]).dt.days
            df.drop(columns=[col], inplace=True)

        # ======================
        # 4. Tratamento de valores ausentes
        # ======================

        # Numéricas
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Categóricas
        cat_cols = df.select_dtypes(include=["object"]).columns
        df[cat_cols] = df[cat_cols].fillna("DESCONHECIDO")

        # ======================
        # 5. One-Hot Encoding
        # ======================

        df = pd.get_dummies(df, drop_first=True)

        X = df

        logger.info(f"Pré-processamento finalizado. X: {X.shape}, y: {y.shape}")
        logger.info(f"Total de NaNs restantes: {X.isna().sum().sum()}")

        return X, y

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        raise