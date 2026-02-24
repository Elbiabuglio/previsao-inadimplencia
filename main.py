from src.database import load_training_data

if __name__ == "__main__":
    df = load_training_data()

    print("Shape:", df.shape)
    print(df.head())