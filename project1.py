import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix;
import matplotlib.pyplot as plt
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
        row['study_hours']* 10 +
        row['attendance']*1+
        row['Quiz_score']*1+
        row['assignment_marks']*1
    )/4

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

X=df[['study_hours','attendance','Quiz_score','assignment_marks']]
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
plt.bar(feature_importance['feature'],feature_importance['importance'])
plt.xlabel('importance score')
plt.title('which Factors Matter Most For Student Preformance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print('Saved Features_Importance.png')

print("\n saving the model")

with open('Student_risk_model.pkl','wb')as file:
    pickle.dump(model,file)
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

print("model saved as 'student_risk_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("\n\n PREDICTION EXAMPLE:")
print('='*50)

def predict_student_risk(study_hours, attendance, Quiz_score,assignment_marks):
    """
     Predict risk level for a new student
    
    Args:
        study_hours: Daily study hours (0-10)
        attendance: Attendance percentage (0-100)
        quiz_scores: Average quiz score (0-100)
        assignment_marks: Average assignment_marks (0-100)
    
    Returns:
        risk_level: 'High Risk', 'Medium Risk', or 'Safe'
        probabilities: Confidence for each risk level
    """
    with open('student_risk_model.pkl','rb')as file:
        loaded_model=pickle.load(file)
    with open('scaler.pkl','rb')as file:
        loaded_scaler=pickle.load(file)
    input_data=np.array([[study_hours,attendance,Quiz_score,assignment_marks]])
    input_scaled=loaded_scaler.transform(input_data)
    
    prediction=loaded_model.predict(input_scaled)[0]
    probabilities=loaded_model.predict_proba(input_scaled)[0]

    return prediction,probabilities
examples = [
    {"name": "Student A (Good)", "study_hours": 6, "attendance": 90, "quiz": 85, "assignment": 88},
    {"name": "Student B (Average)", "study_hours": 3, "attendance": 70, "quiz": 60, "assignment": 65},
    {"name": "Student C (Struggling)", "study_hours": 1, "attendance": 50, "quiz": 40, "assignment": 45},
]

for student in examples:
    risk, probs = predict_student_risk(
        student['study_hours'], 
        student['attendance'], 
        student['quiz'], 
        student['assignment']
    )
    
    print(f"\n{student['name']}:")
    print(f"  Study Hours: {student['study_hours']}h/day")
    print(f"  Attendance: {student['attendance']}%")
    print(f"  Quiz Score: {student['quiz']}")
    print(f"  Assignment: {student['assignment']}")
    print(f"  Prediction: {risk}")
    print(f"  Confidence: {max(probs)*100:.1f}%")











