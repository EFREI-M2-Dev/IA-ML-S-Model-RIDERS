import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("titanic.csv")

data = data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

X = data.drop("Survived", axis=1)
y = data["Survived"]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(classification_report(y_test, y_pred))

scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
