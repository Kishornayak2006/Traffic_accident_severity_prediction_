# 🚦 Traffic Accident Severity Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Active-green)

## 🌐 Live Demo

Try the deployed application here:

🔗 https://your-app-name.streamlit.app


## Web Application: Built a web application using Streamlit and deployed on Streamlit. 
<img width="800" alt="image" src="doc/theme.png">

---

# 📌 Project Overview

Road traffic accidents are one of the major causes of injuries and fatalities worldwide. Predicting accident severity can help authorities and policymakers take preventive measures and improve road safety.

This project builds a **Machine Learning model using XGBoost** to predict the **severity of road traffic accidents** based on multiple environmental, vehicle, and driver-related factors.

The model is deployed as an **interactive web application using Streamlit**.

---

# 🧠 Problem Statement

This is a **Multi-Class Classification Problem** where we predict accident severity into three categories:

• Slight Injury
• Serious Injury
• Fatal Injury

based on **31 input features** such as:

* Time of accident
* Day of week
* Driver age
* Driving experience
* Road surface type
* Weather conditions
* Type of collision
* Number of vehicles involved
* Number of casualties
* Vehicle movement

---

# 📊 Dataset

Dataset Source:
Kaggle – **Road Traffic Accidents Dataset**

The dataset contains **12,316 accident records** with **32 features**.

### Class Distribution

| Severity       | Records |
| -------------- | ------- |
| Slight Injury  | 10,415  |
| Serious Injury | 1,743   |
| Fatal Injury   | 158     |

The dataset is **highly imbalanced**, which required special preprocessing techniques.

---

# 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand:

* Accident patterns
* Driver behavior
* Environmental factors affecting accident severity
* Feature correlations

Visualization techniques helped identify key contributing factors.

---

# 🧹 Data Preprocessing

The dataset contained several missing values and categorical variables.

Steps performed:

• Handling missing values using **Predictive Imputation**
• Label encoding categorical features
• Removing highly correlated features
• Feature scaling and transformation

---

# ⚖️ Handling Data Imbalance

Since the dataset was highly imbalanced, the following techniques were applied:

• **SMOTE (Synthetic Minority Oversampling Technique)**
• Random Over Sampling
• Random Under Sampling
• NearMiss Method

This helped improve model performance on minority classes.

---

# 🎯 Feature Selection

Feature selection was performed using:

• **Correlation Analysis**
• **Chi-Square Test**

Highly correlated features were removed to avoid multicollinearity.

Finally, **16 most important features** were used for model training.

---

# 🤖 Model Training

Several machine learning algorithms were tested:

• Decision Tree
• Random Forest
• Extra Trees
• XGBoost

Among these, **XGBoost performed the best**.

Techniques used:

• **5-Fold Cross Validation**
• **GridSearchCV for Hyperparameter Tuning**

Final model achieved:

🎯 **Accuracy: ~74%**

Evaluation metric used:

**F1 Score**

---

# 🔎 Explainable AI

To understand the model predictions, **SHAP (SHapley Additive Explanations)** was used for interpretability.

This helps explain:

* Which features contribute most to accident severity
* How each feature influences predictions

---

# 🌐 Web Application

An interactive **Streamlit web application** was developed to allow users to input accident conditions and predict severity instantly.

Users can:

• Enter accident details
• Submit the form
• Get real-time accident severity prediction

---

# 🛠️ Tech Stack

### Programming Language

Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib

### Framework

Streamlit

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Kishornayak2006/Traffic_accident_severity_prediction_.git
```

Navigate to the project folder:

```
cd Traffic_accident_severity_prediction_
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit application:

```
streamlit run app.py
```

---

# 📂 Project Structure

```
Traffic_accident_severity_prediction_
│
├── app.py
├── prediction.py
├── config.py
├── requirements.txt
│
├── model
│   ├── model_xgb.bin
│   └── checkpoint.pkl
│
├── doc
│   └── theme.png
│
└── README.md
```

---

# 🚀 Future Improvements

• Improve model performance with **advanced feature engineering**
• Add **visual analytics dashboard**
• Implement **real-time accident risk prediction**

---

# 👨‍💻 Author

**Kishor C**

GitHub:
https://github.com/Kishornayak2006

---

# ⭐ If you like this project

Give this repository a ⭐ on GitHub!
