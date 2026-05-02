import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Create Synthetic Training Data
# Features: [Fever, Cough, Age_Over_60, In_Red_Zone] (1 = Yes, 0 = No)
data = {
    'fever':       [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'cough':       [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
    'age_60':      [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'red_zone':    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    'risk_level':  ['High', 'Low', 'High', 'Low', 'High', 'Medium', 'Low', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)

# 2. Initialize and Train the Model
X = df[['fever', 'cough', 'age_60', 'red_zone']] # Inputs
y = df['risk_level']                            # Target

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. Predict for a NEW User
# Example: User has Fever (1), No Cough (0), Is over 60 (1), and is in Red Zone (1)
new_user = [[1, 0, 1, 1]]
prediction = model.predict(new_user)
probability = model.predict_proba(new_user)

print("--- AI Risk Assessment ---")
print(f"Predicted Risk Level: {prediction[0]}")
print(f"Confidence Score: {max(probability[0])*100:.2f}%")