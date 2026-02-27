import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame):
    """
    Realiza o pré-processamento dos dados para modelagem.

    Etapas:
    1. Criação de variáveis derivadas (engenharia de features)
    2. Seleção de colunas relevantes
    3. Tratamento da variável alvo
    4. Tratamento de datas
    5. Tratamento de valores ausentes
    6. One-Hot Encoding para variáveis categóricas
    7. Separação de preditoras (X) e target (y)
    """

    try:
        logger.info("Iniciando pré-processamento...")
        df = df.copy()

        # =====================================================
        # 1. Criação de variáveis derivadas
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
        # 2. Seleção de colunas relevantes
        # =====================================================
        colunas_relevantes = [
            'TAXA_AO_ANO',
            'CIDADE_CLIENTE',
            'ESTADO_CLIENTE',
            'RENDA_MENSAL_CLIENTE',
            'QT_PC_ATRASO',
            'QT_DIAS_PRIM_PC_ATRASO',
            'QT_TOTAL_PC_PAGAS',
            'VL_TOTAL_PC_PAGAS',
            'QT_PC_PAGA_EM_DIA',
            'QT_DIAS_MIN_ATRASO',
            'QT_DIAS_MAX_ATRASO',
            'QT_DIAS_MEDIA_ATRASO',
            'VALOR_PARCELA',
            'IDADE_DATA_ASSINATURA_CONTRATO',
            'FAIXA_VALOR_FINANCIADO',
            'FAIXA_PRAZO_FINANCIAMENTO',
            'INADIMPLENTE_COBRANCA'
        ]

        colunas_faltantes = set(colunas_relevantes) - set(df.columns)
        if colunas_faltantes:
            logger.warning(f"Colunas ausentes ignoradas: {colunas_faltantes}")

        colunas_existentes = [col for col in colunas_relevantes if col in df.columns]
        df = df[colunas_existentes]

        # =====================================================
        # 3. Tratamento da variável alvo
        # =====================================================
        if "INADIMPLENTE_COBRANCA" not in df.columns:
            raise ValueError("Variável alvo não encontrada no dataset.")

        df["INADIMPLENTE_COBRANCA"] = (
            df["INADIMPLENTE_COBRANCA"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map({"SIM": 1, "NAO": 0})
        )

        if df["INADIMPLENTE_COBRANCA"].isna().sum() > 0:
            raise ValueError("Valores inesperados encontrados na variável alvo.")

        y = df["INADIMPLENTE_COBRANCA"]
        df.drop(columns=["INADIMPLENTE_COBRANCA"], inplace=True)

        # =====================================================
        # 4. Tratamento de datas
        # =====================================================
        hoje = datetime.today()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
        for col in datetime_cols:
            df[f"{col}_dias"] = (hoje - df[col]).dt.days
            df.drop(columns=[col], inplace=True)

        # =====================================================
        # 5. Tratamento de valores ausentes
        # =====================================================
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df[cat_cols] = df[cat_cols].fillna("DESCONHECIDO")

        # =====================================================
        # 6. One-Hot Encoding (somente preditoras categóricas)
        # =====================================================
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # =====================================================
        # 7. Separação final de preditoras (X) e target (y)
        # =====================================================
        X = df.copy()

        logger.info(f"Pré-processamento finalizado. X: {X.shape}, y: {y.shape}")
        logger.info(f"Total de NaNs restantes: {X.isna().sum().sum()}")

        return X, y

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        raise