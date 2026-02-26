from src.database import load_training_data
from src.preprocessing import preprocess_data

df = load_training_data()
df = preprocess_data(df)

df.head()