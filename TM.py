import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv("/Users/roshan/Downloads/Airplane_Crashes_and_Fatalities_Since_1908_20190820105639 (1).csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows with missing critical values
df = df.dropna(subset=['Aboard', 'Fatalities'])

# Create new feature: Severity (1 = severe, 0 = not severe)
df['Severity'] = (df['Fatalities'] / df['Aboard']) > 0.5
df['Severity'] = df['Severity'].astype(int)

# Select features
features = ['Operator', 'AC Type', 'Location', 'Route', 'Aboard', 'Ground']
df = df[features + ['Severity']].dropna()

# Encode categorical variables
for col in ['Operator', 'AC Type', 'Location', 'Route']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split data
X = df[features]
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("aircrash_model.pkl", "wb") as f:
    pickle.dump(model, f)
