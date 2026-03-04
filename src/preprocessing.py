import logging
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

FEATURES_PATH = Path("models/feature_columns.pkl")


def preprocess_data(df: pd.DataFrame, training=True):
    """
    Pré-processamento robusto para treino e produção.
    """

    try:
        logger.info("Iniciando pré-processamento...")
        df = df.copy()

        # =====================================================
        # 1. Feature Engineering
        # =====================================================
        if 'VALOR_FINANCIAMENTO' in df.columns:
            df['FAIXA_VALOR_FINANCIADO'] = pd.cut(
                df['VALOR_FINANCIAMENTO'],
                bins=[0, 5000, 10000, 20000, 50000, float("inf")],
                labels=['Muito_Baixo', 'Baixo', 'Medio', 'Alto', 'Muito_Alto']
            )

        if 'PRAZO_FINANCIAMENTO' in df.columns:
            df['FAIXA_PRAZO_FINANCIAMENTO'] = pd.cut(
                df['PRAZO_FINANCIAMENTO'],
                bins=[0, 12, 24, 48, 72, float("inf")],
                labels=['Curto', 'Medio', 'Longo', 'Muito_Longo', 'Extenso']
            )

        # =====================================================
        # 2. Target (somente no treino)
        # =====================================================
        if training:
            if "INADIMPLENTE_COBRANCA" not in df.columns:
                raise ValueError("Variável alvo não encontrada.")

            df["INADIMPLENTE_COBRANCA"] = (
                df["INADIMPLENTE_COBRANCA"]
                .astype(str)
                .str.upper()
                .str.strip()
                .map({"SIM": 1, "NAO": 0})
            )

            y = df["INADIMPLENTE_COBRANCA"]
            df.drop(columns=["INADIMPLENTE_COBRANCA"], inplace=True)

        else:
            y = None

        # =====================================================
        # 3. Datas
        # =====================================================
        hoje = datetime.today()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns

        for col in datetime_cols:
            df[f"{col}_dias"] = (hoje - df[col]).dt.days
            df.drop(columns=[col], inplace=True)

        # =====================================================
        # 4. Missing Values
        # =====================================================
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df[cat_cols] = df[cat_cols].fillna("DESCONHECIDO")

        # =====================================================
        # 5. One-Hot Encoding
        # =====================================================
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # =====================================================
        # 6. Garantir mesmas colunas do treino
        # =====================================================
        if training:
            FEATURES_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(df.columns.tolist(), FEATURES_PATH)
            logger.info("Colunas salvas para uso em produção.")

        else:
            if not FEATURES_PATH.exists():
                raise FileNotFoundError("Arquivo de colunas do treino não encontrado.")

            training_columns = joblib.load(FEATURES_PATH)

            for col in training_columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[training_columns]

        logger.info(f"Pré-processamento concluído. Shape final: {df.shape}")

        return df, y

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        raise