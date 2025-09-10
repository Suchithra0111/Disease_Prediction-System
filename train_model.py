import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle 
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('dataset1.csv')
df = df.fillna(0)
df = df.dropna(subset=["disease"])

# Use ALL columns except the target
X = df.drop("disease", axis=1)
y = df["disease"].astype(str)
le = LabelEncoder()
y = le.fit_transform(y)
# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save both model + feature names
feature_names = X.columns.tolist()
with open("model.pkl", "wb") as f:
    pickle.dump((model, feature_names, le), f)
print("✅ Model trained on", len(feature_names), "features and saved as model.pkl")
