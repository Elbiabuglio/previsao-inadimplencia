from src.database import load_training_data
from src.preprocessing import preprocess_data
from src.modeling import train_model, evaluate_model, save_model

if __name__ == "__main__":
    df = load_training_data()
    X, y = preprocess_data(df)

    # Treinamento com balanceamento SMOTE
    model, X_test, y_test = train_model(X, y, apply_smote=True)
    evaluate_model(model, X_test, y_test)

    save_model(model)