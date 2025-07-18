import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv(r'D:\loan-prediction-flask-app\Dataset\train.csv')
test = pd.read_csv(r'D:\loan-prediction-flask-app\Dataset\test.csv')

# Save target variable and Loan_ID
Loan_status = train['Loan_Status'].map({'Y': 1, 'N': 0})
Loan_ID = test['Loan_ID']

# Drop target from train set
train.drop(['Loan_Status'], axis=1, inplace=True)

# Combine train + test
data = pd.concat([train, test], ignore_index=True)

# Drop Loan_ID before correlation
data.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical values
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Rural': 0, 'Semiurban': 1})

# Fill missing values
data['Credit_History'].fillna(np.random.randint(0, 2), inplace=True)
data['Married'].fillna(np.random.randint(0, 2), inplace=True)
data['Gender'].fillna(np.random.randint(0, 2), inplace=True)
data['Dependents'].fillna(data['Dependents'].median(), inplace=True)
data['Self_Employed'].fillna(np.random.randint(0, 2), inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)

# 🧠 Correlation matrix (on numeric values only)
numeric_data = data.select_dtypes(include=[np.number])
corrmat = numeric_data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)
plt.show()

# Split back to train & test
train_X = data.iloc[:614, :]
X_test = data.iloc[614:, :]
train_y = Loan_status

# Train/Test split for model evaluation
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, random_state=7)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score

# Define models
models = [
    ("logreg", LogisticRegression()),
    ("tree", DecisionTreeClassifier()),
    ("lda", LinearDiscriminantAnalysis()),
    ("svc", SVC()),
    ("knn", KNeighborsClassifier()),
    ("nb", GaussianNB())
]

# Evaluate models
print("Model Accuracy Results:\n")
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_result = cross_val_score(model, train_X, train_y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_result.mean():.4f}, Std = {cv_result.std():.4f}")

# Train final model
final_model = LogisticRegression()
final_model.fit(train_X, train_y)

# Evaluate on test set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pred = final_model.predict(test_X)
print("\nTest Set Evaluation:\n")
print("Accuracy:", accuracy_score(test_y, pred))
print("Confusion Matrix:\n", confusion_matrix(test_y, pred))
print("Classification Report:\n", classification_report(test_y, pred))

# Predict for actual test data
outp = final_model.predict(X_test).astype(int)

# Create output DataFrame
df_output = pd.DataFrame({'Loan_ID': Loan_ID, 'Loan_Status': outp})
df_output['Loan_Status'] = df_output['Loan_Status'].map({1: 'Y', 0: 'N'})  # Optional: match original format

# Export
df_output.to_csv(r'D:\loan-prediction-flask-app\Dataset\output.csv', index=False)

import joblib
joblib.dump(final_model, 'loan_model.pkl')
# Save the model for future use