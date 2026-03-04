import logging
import sys

from src.database import (
    test_connection,
    load_training_data,
    save_predictions,   # <- substitui insert_dataframe
    execute_procedure
)

from src.preprocessing import preprocess_data
from src.modeling import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict
)

# ==========================================
# CONFIGURAÇÃO DE LOG
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# PIPELINE DE TREINAMENTO
# ==========================================
def run_training_pipeline():
    logger.info("Iniciando pipeline de treinamento...")

    df = load_training_data()
    X, y = preprocess_data(df, training=True)  # <-- garante pré-processamento para treino

    model, X_test, y_test = train_model(X, y, apply_smote=True)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    logger.info("Pipeline de treinamento finalizado com sucesso.")

# ==========================================
# PIPELINE DE SCORING (PRODUÇÃO)
# ==========================================
def run_scoring_pipeline():
    logger.info("Iniciando pipeline de scoring...")

    df = load_training_data()

    model = load_model()

    # Pré-processamento sem tocar a variável alvo
    X, _ = preprocess_data(df, training=False)

    # Gera predições
    df_resultado = predict(model, X, df)

    # Salva previsões no banco
    save_predictions(df_resultado, table_name="RESULTADOS_INTERMEDIARIOS")

    # Executa procedure final
    execute_procedure("SP_INPUT_RESULTADOS_MODELO_PREDITIVO")

    logger.info("Pipeline de scoring finalizado com sucesso.")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    try:
        logger.info("Validando conexão com banco...")
        if not test_connection():
            logger.error("Falha na conexão com banco.")
            sys.exit(1)

        # 🔥 Escolha qual pipeline rodar
        MODE = "score"  

        if MODE == "train":
            run_training_pipeline()
        elif MODE == "score":
            run_scoring_pipeline()
        else:
            logger.error("Modo inválido. Use 'train' ou 'score'.")

    except Exception as e:
        logger.exception(f"Erro crítico na execução: {e}")
        sys.exit(1)