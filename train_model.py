import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')
X = df.drop('disease', axis=1)
y = df['disease']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model trained and saved as model.pkl")