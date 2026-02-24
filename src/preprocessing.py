import pandas as pd


def remove_colunas_valor_unico(df):

    nunique = df.nunique()
    colunas_unicas = nunique[nunique == 1].index
    df = df.drop(columns=colunas_unicas)

    return df


def criar_faixa_valor_financiamento(df):

    df["FAIXA_VALOR_FINANCIAMENTO"] = pd.cut(
        df["VALOR_FINANCIAMENTO"],
        bins=[0, 5000, 10000, 20000, 50000, 100000],
        labels=["0-5k", "5k-10k", "10k-20k", "20k-50k", "50k+"]
    )

    return df