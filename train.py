import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Handle missing and non-numeric TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Define features and target
features = ['tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
            'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
            'StreamingMovies', 'StreamingTV']
target = 'Churn'

# Encode categorical features
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
df[target] = LabelEncoder().fit_transform(df[target])

# Train/test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
os.makedirs("data", exist_ok=True)
with open("data/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to data/model.pkl")
