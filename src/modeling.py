import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path


logger = logging.getLogger(__name__)


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em treino e teste e treina um modelo de Regressão Logística.

    Parameters
    ----------
    X : pd.DataFrame
        Variáveis independentes.
    y : pd.Series
        Variável target.
    test_size : float
        Proporção do conjunto de teste.
    random_state : int
        Seed para reprodutibilidade.

    Returns
    -------
    model : sklearn model
        Modelo treinado.
    X_test : pd.DataFrame
        Dados de teste.
    y_test : pd.Series
        Target de teste.
    """

    logger.info("Iniciando divisão treino/teste...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Treinando modelo Logistic Regression...")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    logger.info("Modelo treinado com sucesso.")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo com métricas de classificação.
    """

    logger.info("Avaliando modelo...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    return auc


def save_model(model, model_name="modelo_inadimplencia.pkl"):
    """
    Salva o modelo treinado na pasta models.
    """

    logger.info("Salvando modelo...")

    models_path = Path("models")
    models_path.mkdir(exist_ok=True)

    model_path = models_path / model_name

    joblib.dump(model, model_path)

    logger.info(f"Modelo salvo em: {model_path}")