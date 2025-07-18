# 🏦 Loan Prediction Web App

A fully responsive and animated Flask web application that predicts whether a loan application will be **Approved ✅** or **Not Approved ❌** based on user inputs.

<img width="1309" height="620" alt="Screenshot 2025-07-18 154149" src="https://github.com/user-attachments/assets/a793a5ed-f12b-42bc-8f97-553c2abc8f0c" />


---

## 🚀 Features

- 🎯 Predicts loan approval using a trained machine learning model
- 🎨 Glassmorphism UI with responsive layout
- 📱 Mobile-first design, perfect on all screens
- 🔥 Live prediction using Logistic Regression
- 🧠 Built with scikit-learn, Flask, HTML, CSS, JavaScript

---

## 🧠 Machine Learning Model

- Model: **Logistic Regression**
- Dataset: Cleaned and preprocessed 
- Accuracy: ✅ Evaluated using K-Fold Cross Validation
- Exported model file: `loan_model.pkl` (saved using `joblib`)

---

## 🖼️ UI Overview

| Input Fields          | Layout                      |
|-----------------------|-----------------------------|
| Gender, Married       | Side-by-side in 1st row     |
| Dependents, Education | Side-by-side in 2nd row     |
| Self Employed, Income | Side-by-side in 3rd row     |
| Co-Income, Loan Amt   | Side-by-side in 4th row     |
| Loan Term, Credit     | Side-by-side in 5th row     |
| Property Area + Btn   | Final row with Predict 🔮   |

> 💎 Glassmorphism form design with animation and hover effects.

---

## 🛠 Tech Stack

| Category      | Tools Used                          |
|---------------|-------------------------------------|
| Backend       | Python, Flask                       |
| ML Model      | scikit-learn, pandas, numpy         |
| Frontend      | HTML, CSS (Glassmorphism), JS       |
| Visualization | Matplotlib, Seaborn                 |
| Deployment    | ⚙️ Ready for Render, Railway, etc.  |

---

## 🗂️ Project Structure
Loan-Prediction-WebApp/
├── app.py
├── loan_model.pkl
├── requirements.txt
├── README.md
├── Dataset/
│   └── train.csv
│   └── test.csv
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    └── style.css







## 🔥 How to Run Locally

1. **Clone the Repo**
   ```bash
   https://github.com/Aaru-0653/Loan-Predictor-Ai-Model
   cd loan-prediction-webapp


## Install Requirements
pip install -r requirements.txt
Run Flask App

python app.py

Open Browser

http://127.0.0.1:5000/

