import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix;
import matplotlib as plt
import seaborn as sns
import pickle

print("Creating sample data")

np.random.seed(42)
n_samples=200

data={
    'study_hours':np.random.uniform(0,10,n_samples),
    'attendance':np.random.uniform(40,100,n_samples),
    'Quiz_score':np.random.uniform(0,100,n_samples),
    'assignment_marks':np.random.uniform(0,100,n_samples),
}
df=pd.DataFrame(data)

def assign_risk(row):
    preformance=(
        row['study_hours']* 5 +
        row['attendance']*0.3+
        row['Quiz_score']*0.4+
        row['assignment marks']*0.3
    )/100

    if preformance<40:
        return "High Risk"
    elif preformance<70:
        return "Medium Risk"
    else:
        return"Safe"
df['risk_level']=df.apply(assign_risk,axis=1)

print(f"created {len(df)} student recordes ")

print(f"\n Risk Distribution")
print(df['risk_level'].value_counts())

print("\n preparing data for training")

X=df[['study_hours','attendance','Qiuz_score','assignment_marks']]
y=df["risk_level"]

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print(f"Training set: {len(X_train)} students")
print(f"Testing set: {len(X_test)} students")

print("\n training model")

model=RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train_scaled,y_train)

print(" Model training complete!")

