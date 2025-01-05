import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Model Prediction",
    page_icon="📈",
)

st.title("Model Evaluation and Prediction")
st.markdown(
    """
📈 **Model Performance and Model Prediction**
"""
)

# Sidebar navigation
st.sidebar.title("Model Evaluation Sections")
section = st.sidebar.radio(
    "Choose a section:",
    ["Model Performance", "Model Prediction"],
)


# Load dummy data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Refdinal/insight-hustler/refs/heads/master/data/raw/aug_train.csv"
    )
    return df


def load_data2():
    df2 = pd.read_csv(
        "https://raw.githubusercontent.com/Refdinal/insight-hustler/refs/heads/master/data/processed/data_preprocessed.csv"
    )
    return df2


def load_data3():
    df3 = pd.read_csv(
        "https://raw.githubusercontent.com/Refdinal/insight-hustler/refs/heads/master/data/processed/model_result.csv"
    )
    return df3


def evaluate_metrics(model, X_train, y_train, X_test, y_test):
    """
    Evaluates a classification model and returns a DataFrame with key metrics for both train and test datasets.

    Parameters:
    - model: Trained classification model (e.g., LogisticRegression, RandomForestClassifier, etc.)
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels

    Returns:
    - metrics_df: A DataFrame containing evaluation metrics for both train and test sets.
    """
    # Predictions and probabilities
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Metrics for Train Set
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_pred_train),
        "Precision": precision_score(y_train, y_pred_train),
        "Recall": recall_score(y_train, y_pred_train),
        "F1-Score": f1_score(y_train, y_pred_train),
        "ROC-AUC": roc_auc_score(y_train, y_proba_train),
    }

    # Metrics for Test Set
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test),
        "Recall": recall_score(y_test, y_pred_test),
        "F1-Score": f1_score(y_test, y_pred_test),
        "ROC-AUC": roc_auc_score(y_test, y_proba_test),
    }

    # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])

    return metrics_df.T


df = load_data()
df2 = load_data2()
df3 = load_data3()
X = df2.drop(columns=["target"])
y = df2["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train_resampled)
X_train_scaled = scaler.transform(X_train_resampled)  # Fit dan transform data latih
X_test_scaled = scaler.transform(X_test)  # Hanya transform data uji


# Section: Model Performance
if section == "Model Performance":

    # Descriptive statistics
    st.header("Model Performance")
    st.write(
        "We have tried several classification tuned models with the following evaluation metrics."
    )
    st.dataframe(df3, use_container_width=True)
    st.markdown(
        """
    # Insights

## 1. Best Performance
- **CatBoost** achieves the highest test accuracy (0.785893) with good stability, making it the most superior model.
- **LGBM Classifier** ranks second with a comparable test accuracy (0.784227).

## 2. Overfitting
- **Random Forest** exhibits potential overfitting with a high train accuracy (0.837521) but lower test accuracy (0.765621).
- **KNN** and **Decision Tree** show significant overfitting with a large gap between train and test accuracy.

## 3. Consistency
- **Logistic Regression** is highly stable between train (0.796162), cross-validation, and test accuracy (0.771730), though its test accuracy is lower compared to more complex models.

## 4. Recommendations
- Use **CatBoost** for the best performance.
- Consider **LGBM Classifier** as a balanced alternative.
- Opt for **Logistic Regression** if simplicity and consistency are prioritized.

"""
    )
# Section: Model Prediction
if section == "Model Prediction":

    # Descriptive statistics
    st.header("Model Prediction")
    with st.form(key="employee_form"):
        city_development_index = st.number_input(
            "City Development Index", min_value=0.0, max_value=1.0, step=0.01
        )
        training_hours = st.number_input("Training Hours", min_value=0, step=1)
        gender = st.selectbox("Gender", options=["NaN", "Male", "Female", "Other"])
        relevant_experience = st.selectbox(
            "Relevant Experience",
            options=["Has relevent experience", "No relevent experience"],
        )
        enrolled_university = st.selectbox(
            "Enrolled University",
            options=[
                "no_enrollment",
                "Full time course",
                "Part time course",
            ],
        )
        education_level = st.selectbox(
            "Education Level",
            options=["Primary School", "High School", "Graduate", "Masters", "Phd"],
        )
        major_discipline = st.selectbox(
            "Major Discipline",
            options=[
                "NaN",
                "STEM",
                "Humanities",
                "Other",
                "Business Degree",
                "Arts",
                "No Major",
            ],
        )
        experience = st.selectbox(
            "Experience",
            options=[
                "<1",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                ">20",
            ],
        )  # -1 for empty option
        company_size = st.selectbox(
            "Company Size",
            options=[
                "NaN",
                "<10",
                "10/49",
                "50-99",
                "100-500",
                "500-999",
                "1000-4999",
                "5000-9999",
                "10000+",
            ],
        )
        company_type = st.selectbox(
            "Company Type",
            options=[
                "NaN",
                "Pvt Ltd",
                "Funded Startup",
                "Public Sector",
                "Early Stage Startup",
                "NGO",
                "Other",
            ],
        )
        last_new_job = st.selectbox(
            "Last New Job",
            options=[
                "never",
                "1",
                "2",
                "3",
                "4",
                ">4",
            ],
        )  # -1 for empty option

        # Submit button
        submit_button = st.form_submit_button("Submit")
    if submit_button:
        data = [
            city_development_index,
            gender,
            relevant_experience,
            enrolled_university,
            education_level,
            major_discipline,
            experience,
            company_size,
            company_type,
            last_new_job,
            training_hours,
        ]
        col = df.columns[2:13]
        df_pred = pd.DataFrame([data], columns=col)

        # Cleansing and handling missing value
        df_pred["last_new_job"] = df_pred["last_new_job"].apply(
            lambda x: "Never" if x == "never" else x
        )  # just reads nicer
        df_pred["enrolled_university"][
            df_pred["enrolled_university"] == "no_enrollment"
        ] = "No Enrollment"  # just reads nicer
        df_pred["company_size"] = df_pred["company_size"].apply(
            lambda x: "10-49" if x == "10/49" else x
        )  # diff replacement method

        df_pred["experience"] = df_pred["experience"].apply(
            lambda x: "0" if x == "<1" else x
        )
        df_pred["experience"] = df_pred["experience"].apply(
            lambda x: "21" if x == ">20" else x
        )
        company_size = np.nan if major_discipline == "NaN" else company_size
        company_type = np.nan if major_discipline == "NaN" else company_type
        major_discipline = np.nan if major_discipline == "NaN" else major_discipline
        gender = np.nan if major_discipline == "Select an option" else gender
        df_pred["company_size"].fillna("0", inplace=True)
        df_pred["company_type"].fillna("Unknown", inplace=True)
        df_pred["major_discipline"].fillna("Unknown", inplace=True)
        df_pred["gender"].fillna("Other", inplace=True)
        # transformasi log
        df_pred["training_hours"] = np.log1p(df_pred["training_hours"])
        # encoding
        df_pred["relevent_experience"] = [
            1 if i == "Has relevent experience" else 0
            for i in df_pred["relevent_experience"]
        ]
        df_pred["education_level"] = [
            (
                5
                if i == "Phd"
                else (
                    4
                    if i == "Masters"
                    else 3 if i == "Graduate" else 2 if i == "High School" else 1
                )
            )
            for i in df_pred["education_level"]
        ]
        df_pred["experience"] = df_pred["experience"].astype(int)
        df_pred["last_new_job"] = [
            (
                5
                if i == ">4"
                else (
                    4
                    if i == "4"
                    else 3 if i == "3" else 2 if i == "2" else 1 if i == "1" else 0
                )
            )
            for i in df_pred["last_new_job"]
        ]
        df_pred = pd.get_dummies(df_pred, columns=["gender"], drop_first=False)
        df_pred = pd.get_dummies(
            df_pred, columns=["enrolled_university"], drop_first=False
        )
        df_pred = pd.get_dummies(
            df_pred, columns=["major_discipline"], drop_first=False
        )
        # buat kategori baru
        df_pred["company_size"] = df_pred["company_size"].apply(
            lambda x: (
                "Small"
                if x in ["<10", "10-49", "50-99"]
                else (
                    "Medium"
                    if x in ["100-500", "500-999"]
                    else "Large" if x in ["1000-4999", "5000-9999"] else "Unknown"
                )
            )  # For '0' or missing values
        )
        df_pred = pd.get_dummies(df_pred, columns=["company_size"], drop_first=False)
        df_pred = pd.get_dummies(df_pred, columns=["company_type"], drop_first=False)
        df_pred = df_pred.reindex(columns=df2.columns, fill_value=0)
        df_pred = df_pred.drop(columns=["target"])
        st.dataframe(df_pred)
        model = load("/mount/src/insight-hustler/apps/models/xgboost.joblib")
        # model = load("./models/xgboost.joblib")
        # Predict probabilities
        pred_result = model.predict(df_pred)

        # Extract probability for the positive class (e.g., column index 1 for binary classification)
        positive_class_prob = pred_result[0][1]

        # Display the result in Streamlit
        st.write("Prediction Probability: ", positive_class_prob)

        # Determine and display the final outcome
        if positive_class_prob > 0.5:
            st.success("Employee is likely to find a new job.")
        else:
            st.info("Employee is likely to stay.")
