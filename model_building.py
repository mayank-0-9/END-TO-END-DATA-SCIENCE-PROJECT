import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# -------- Step 1: Data Loading --------
df = pd.read_csv('data.csv')
print(df.head())

# -------- Step 2: Preprocessing --------
# Let's assume 'Attrition' is the target column; change if yours is different
X = df.drop('Attrition', axis=1)  # Remove target from features
y = df['Attrition']

# Encode categorical variables (e.g., one-hot encoding)
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Step 3: Model Training --------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# -------- Step 4: Evaluation --------
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# -------- Step 5: Save Model, Scaler, and Feature Names --------
joblib.dump(model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')
print("✔️ Model, scaler, and feature names have been saved!")