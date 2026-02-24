from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def treinar_modelo(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred))

    joblib.dump(modelo, "models/modelo_inadimplencia.pkl")

    return modelo