import logging
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE  # <-- Import do SMOTE

logger = logging.getLogger(__name__)


def train_model(X, y, test_size=0.2, random_state=42, apply_smote=True):
    """
    Divide os dados em treino e teste e treina um modelo
    de Regressão Logística com Pipeline e padronização.
    
    Parâmetros:
    -----------
    X : pd.DataFrame
        Variáveis preditoras
    y : pd.Series
        Variável alvo
    test_size : float
        Proporção do conjunto de teste
    random_state : int
        Seed para reprodutibilidade
    apply_smote : bool
        Se True, aplica SMOTE no conjunto de treino
    """

    logger.info("Iniciando divisão treino/teste...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # =====================================================
    # Aplicar SMOTE apenas no conjunto de treino
    # =====================================================
    if apply_smote:
        logger.info("Aplicando SMOTE para balanceamento da variável alvo...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Novo shape após SMOTE: X_train={X_train.shape}, y_train={y_train.shape}")

    # =====================================================
    # Pipeline com padronização + Regressão Logística
    # =====================================================
    logger.info("Treinando modelo Logistic Regression com Pipeline...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    model.fit(X_train, y_train)

    logger.info("Modelo treinado com sucesso.")

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, threshold=0.4):
    """
    Avalia o modelo com ajuste de threshold manual.
    """

    logger.info("Avaliando modelo...")

    y_prob = model.predict_proba(X_test)[:, 1]

    # Ajuste de threshold manual
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)

    logger.info(f"Threshold utilizado: {threshold}")
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