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

print("/n Evaluating model performance")

y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(y_test, y_pred)
print(f"\n Model accuracy is: {accuracy *100:.2f}%")

print("\n Detailed performance report")
print(classification_report(y_test,y_pred))

feature_importance=pd.DataFrame({
    'feature':X.columns,
    'importance':model.feature_importances_
}).sort_values('importance',ascending=False)

print("\n Feature Importance (what matters most):")
print(feature_importance)

print("\n creating visualization")
plt.figure(figsize=(8,6))

cm=confusion_matrix(y_test,y_pred,labels=['High Risk','Medium Risk','Safe'])
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues',
            xticklabels=['High Risk','Medium Risk','Safe'],
            yticklabels=['High Risk','Medium Risk','Safe'])
plt.title('Confusion Matrix - Prediction Accuracy')
plt.ylabel('Actual Risk Level')
plt.xlabel('Predicted risk level')
plt.tight_layout()
plt.savefig('Confution_matrix.png')

print("Saved confusion_matrix.png")

plt.figure(figsize=(10,6))
plt.brah(feature_importance['feature'],feature_importance['importance'])
plt.xlabel('importance score')
plt.title('which Factors Matter Most For Student Preformance')
plt.tight_layout()
plt.savegif('feature_importance.png')
print('Saved Features_Importance.png')

print("\n saving the model")







