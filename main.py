from src.database import load_training_data
from src.preprocessing import preprocess_data

if __name__ == "__main__":
    df = load_training_data()
    X, y = preprocess_data(df)

    print("X shape:", X.shape)
    print("y shape:", y.shape)