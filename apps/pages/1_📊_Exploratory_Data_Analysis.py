import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
)

st.title("Exploratory Data Analysis (EDA)")
st.markdown(
    "📊 **Analyze the dataset to understand key patterns and trends in employee attrition.**"
)

# Sidebar navigation
st.sidebar.title("EDA Sections")
section = st.sidebar.radio(
    "Choose a section:",
    [
        "Dataset Overview",
        "Data Cleansing",
        "Feature Transformation",
        "Feature Encoding",
        "Feature Engineering",
        "Split Data Train-Test",
        "Imbalance Data",
        "Scaling Data",
    ],
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


df = load_data()
df2 = load_data2()
nums = ["enrollee_id", "city_development_index", "training_hours"]
cats = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "experience",
    "company_size",
    "company_type",
    "last_new_job",
]
target = ["target"]

# Section: Dataset Overview
if section == "Dataset Overview":

    # Descriptive statistics
    st.header("Dataset Overview")
    st.write("---")
    st.subheader("Dataset Description")
    st.text("Dataset has 3 numerical features, 10 categorical features, and 1 target.")
    num_desc = {
        "Feature Name": [
            "enrolle_id",
            "city_development_index",
            "training_hours",
            "city",
            "gender",
            "relevent_experience",
            "enrolled_university",
            "education_level",
            "major_discipline",
            "experience",
            "company_size",
            "company_type",
            "last_new_job",
            "target",
        ],
        "Description": [
            "A unique ID assigned to each employee",
            "The City Development Index associated with the employee's location",
            "The total number of training hours completed by the employee",
            "The city where the employee is located",
            "The gender of the employee",
            "Indicates whether the employee has relevant experience in the field",
            "The university the employee enrolled in (if any)",
            "The education level of the employee",
            "The major discipline of the employee's studies",
            "The number of years of experience the employee has",
            "The size of the company where the employee works",
            "The type of company the employee works for",
            "The number of months since the employee's last job change",
            "1 if the employee is leaving the job, and 0 if staying",
        ],
    }

    num_df = pd.DataFrame(num_desc).reset_index(drop=True)
    st.table(num_df)
    st.write("Numerical Feature Overview")
    st.dataframe(df[nums].describe().T, use_container_width=True)
    st.write("Categorical Feature Overview")
    st.dataframe(df[cats].describe().T, use_container_width=True)
    st.write("Target Overview")
    st.dataframe(df[target].describe().T, use_container_width=True)

# Section: Data Cleansing
elif section == "Data Cleansing":
    st.header("Data Cleansing")
    st.text(f"We have {df.shape[0]} data before Cleansing")
    st.subheader("Missing Values")
    st.dataframe(df.isna().sum(), use_container_width=True)
    st.write(
        "There are many missing values in categorical features that are valuable for analysis and modeling, so we decided to replace them with a new category, such as 'Unknown' and '0', instead of removing them. "
    )
    df["last_new_job"] = df["last_new_job"].apply(
        lambda x: "Never" if x == "never" else x
    )  # just reads nicer
    df["enrolled_university"][
        df["enrolled_university"] == "no_enrollment"
    ] = "No Enrollment"  # just reads nicer
    df["company_size"] = df["company_size"].apply(
        lambda x: "10-49" if x == "10/49" else x
    )  # diff replacement method

    df["experience"] = df["experience"].apply(lambda x: "0" if x == "<1" else x)
    df["experience"] = df["experience"].apply(lambda x: "21" if x == ">20" else x)

    df["company_size"].fillna("0", inplace=True)
    df["company_type"].fillna("Unknown", inplace=True)
    df["major_discipline"].fillna("Unknown", inplace=True)
    df["gender"].fillna("Other", inplace=True)
    df.dropna(inplace=True)
    st.write(
        f"Some features have missing values that are less than 2%, and they will be removed immediately. Therefore, the number of records after handling missing values is {df.shape[0]}."
    )
    st.subheader("Duplicated Data")
    st.write(
        f"There is no duplicate data, and the number of records remains the same {df.shape[0]}."
    )
    st.subheader("Outlier")
    st.write(
        "We are only checking for outliers in the city_development_index and training_hours. "
    )
    outlier_to_check = ["city_development_index", "training_hours"]
    fig, axes = plt.subplots(len(outlier_to_check), 1, figsize=(10, 10))
    for i, feature in enumerate(outlier_to_check):
        sns.boxplot(x="target", y=feature, data=df, ax=axes[i])
        axes[i].set_title(f"{feature} vs target (Before Handling Outliers)")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    st.write(
        "city_development_index : There are few outliers, but they may have important meaning in relation to the target (city conditions can influence job search behavior)."
    )
    st.write(
        f"We decided not to remove the outliers earlier and will do so after some data transformation. The dataset contains {df.shape[0]} rows."
    )

# Section: Feature Transformation
elif section == "Feature Transformation":

    df["last_new_job"] = df["last_new_job"].apply(
        lambda x: "Never" if x == "never" else x
    )  # just reads nicer
    df["enrolled_university"][
        df["enrolled_university"] == "no_enrollment"
    ] = "No Enrollment"  # just reads nicer
    df["company_size"] = df["company_size"].apply(
        lambda x: "10-49" if x == "10/49" else x
    )  # diff replacement method

    df["experience"] = df["experience"].apply(lambda x: "0" if x == "<1" else x)
    df["experience"] = df["experience"].apply(lambda x: "21" if x == ">20" else x)

    df["company_size"].fillna("0", inplace=True)
    df["company_type"].fillna("Unknown", inplace=True)
    df["major_discipline"].fillna("Unknown", inplace=True)
    df["gender"].fillna("Other", inplace=True)
    df.dropna(inplace=True)

    st.header("Feature Transformation")
    st.subheader("Numeric Feature Transformation")
    st.text("city_development_index feature")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=df[df["target"] == 0]["city_development_index"],
        label="Target 0",
        shade=True,
    )
    sns.kdeplot(
        data=df[df["target"] == 1]["city_development_index"],
        label="Target 1",
        shade=True,
    )
    plt.title("KDE Plot: City Development Index by Target")
    plt.xlabel("City Development Index")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    st.pyplot(plt)
    st.write(
        "We didn't perform any transformation or remove outliers for this feature because the outliers have a strong explanation for differentiating between the 1 and 0 target."
    )
    st.text("training_hours feature")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=df[df["target"] == 0]["training_hours"], label="Target 0", shade=True
    )
    sns.kdeplot(
        data=df[df["target"] == 1]["training_hours"], label="Target 1", shade=True
    )
    plt.title("KDE Plot: Training Hours by Target (Before Log Transformation)")
    plt.xlabel("Training Hours")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    st.pyplot(plt)
    df["training_hours"] = np.log1p(df["training_hours"])
    st.write("Applying Log transformation to this feature :")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=df[df["target"] == 0]["training_hours"], label="Target 0", shade=True
    )
    sns.kdeplot(
        data=df[df["target"] == 1]["training_hours"], label="Target 1", shade=True
    )
    plt.title("KDE Plot: Training Hours by Target (After Log Transformation)")
    plt.xlabel("Training Hours")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    st.pyplot(plt)
    st.write("Remove the outlier with zscore 3")
    # Menyimpan shape data sebelum penghapusan outlier
    df_before = df.shape[0]
    # Menghapus outlier berdasarkan z-score
    df = df[(zscore(df["training_hours"]) > -3) & (zscore(df["training_hours"]) < 3)]
    st.write(f"Data records before transformation = {df_before}")
    st.write(f"Data records after transformation = {df.shape[0]}")

# Section: Feature Encoding
elif section == "Feature Encoding":

    df["last_new_job"] = df["last_new_job"].apply(
        lambda x: "Never" if x == "never" else x
    )  # just reads nicer
    df["enrolled_university"][
        df["enrolled_university"] == "no_enrollment"
    ] = "No Enrollment"  # just reads nicer
    df["company_size"] = df["company_size"].apply(
        lambda x: "10-49" if x == "10/49" else x
    )  # diff replacement method

    df["experience"] = df["experience"].apply(lambda x: "0" if x == "<1" else x)
    df["experience"] = df["experience"].apply(lambda x: "21" if x == ">20" else x)

    df["company_size"].fillna("0", inplace=True)
    df["company_type"].fillna("Unknown", inplace=True)
    df["major_discipline"].fillna("Unknown", inplace=True)
    df["gender"].fillna("Other", inplace=True)
    df.dropna(inplace=True)
    df["training_hours"] = np.log1p(df["training_hours"])

    df = df[(zscore(df["training_hours"]) > -3) & (zscore(df["training_hours"]) < 3)]

    st.header("Feature Encoding")
    st.write("---")
    st.subheader("Label Encoding")
    st.markdown(
        """

    Label Encoding is applied to:
    1. Categorical features with ordinal data type (can be ranked or ordered).
    2. Features that have only two unique categorical values.
    """
    )
    st.write("---")
    st.text("relevent_experience")
    st.write("labelled 1 if has relevent experience and 0 if no relevent experience")
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        sns.countplot(x="relevent_experience", data=df)
        plt.title("relevent_experience before label encoding")
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the first plot

    # Label encoding
    df["relevent_experience"] = [
        1 if i == "Has relevent experience" else 0 for i in df["relevent_experience"]
    ]

    # Second plot: After Label Encoding
    with col2:
        plt.figure(figsize=(8, 6))
        sns.countplot(x="relevent_experience", data=df)
        plt.title("relevent_experience after label encoding")
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the second plot
    st.write("---")

    st.text("education level")
    st.write(
        "labelled from 1 to 5 base on education level in order, primary school, High School, Bachelors/Graduate,Masters,PhD"
    )
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("education_level before label encoding")
        sns.countplot(x="education_level", data=df)
        plt.xticks(rotation=30)
        st.pyplot(plt)  # Display the first plot

    # Label encoding
    df["education_level"] = [
        (
            5
            if i == "Phd"
            else (
                4
                if i == "Masters"
                else 3 if i == "Graduate" else 2 if i == "High School" else 1
            )
        )
        for i in df["education_level"]
    ]

    # Second plot: After Label Encoding
    with col2:
        plt.figure(figsize=(8, 6))
        plt.title("education_level after label encoding")
        sns.countplot(x="education_level", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the second plot
    st.write("---")
    st.text("experience")
    st.write(
        "The experience data type is currently a string. We will convert it into an integer, and the order will become clear."
    )
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("experience before label encoding")
        sns.countplot(x="experience", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the first plot

    # Label encoding
    df["experience"] = df["experience"].astype(int)

    # Second plot: After Label Encoding
    with col2:
        plt.figure(figsize=(8, 6))
        plt.title("experience after label encoding")
        sns.countplot(x="experience", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the second plot
    st.write("---")
    st.text("last_new_job")
    st.write(
        "Similar to the experience feature above, the last_new_job data will be converted into integers so that its order can be clearly observed."
    )
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("last_new_job before label encoding")
        sns.countplot(x="last_new_job", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the first plot

    # Label encoding
    df["last_new_job"] = [
        (
            5
            if i == ">4"
            else (
                4
                if i == "4"
                else 3 if i == "3" else 2 if i == "2" else 1 if i == "1" else 0
            )
        )
        for i in df["last_new_job"]
    ]

    # Second plot: After Label Encoding
    with col2:
        plt.figure(figsize=(8, 6))
        plt.title("last_new_job after label encoding")
        sns.countplot(x="last_new_job", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the second plot
    st.write("---")
    st.write("---")
    st.subheader("One-Hot Encoding")
    st.markdown(
        """One-hot encoding dilakukan pada categorical data yang bukan tipe data ordinal(tidak dapat diurutkan)"""
    )
    st.write("---")
    st.text("gender")
    st.write(
        "In the previous stage, missing values in the gender column were filled with 'other'. One-Hot Encoding applied to this feature "
    )
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("gender distribution")
        sns.countplot(x="gender", data=df)
        plt.xticks(rotation=60)
        st.pyplot(plt)  # Display the first plot
    with col2:
        df = pd.get_dummies(df, columns=["gender"], drop_first=False)
        st.dataframe(df.reset_index().sample(10).T.tail(3).T, use_container_width=True)
    st.write("---")
    st.text("enrolled_university")
    st.write("One-Hot Encoding applied to this feature ")
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("enrolled_university distribution")
        sns.countplot(x="enrolled_university", data=df)
        plt.xticks(rotation=0)
        st.pyplot(plt)  # Display the first plot
    with col2:
        df = pd.get_dummies(df, columns=["enrolled_university"], drop_first=False)
        st.dataframe(df.reset_index().sample(10).T.tail(3).T, use_container_width=True)
    st.write("---")
    st.text("major_discipline")
    st.write("One-Hot Encoding applied to this feature ")
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("major_discipline distribution")
        sns.countplot(x="major_discipline", data=df)
        plt.xticks(rotation=30)
        st.pyplot(plt)  # Display the first plot
    with col2:
        df = pd.get_dummies(df, columns=["major_discipline"], drop_first=False)
        st.dataframe(df.reset_index().sample(10).T.tail(7).T, use_container_width=True)
    st.write("---")
    st.text("company_size")
    st.write(
        "We narrowed down the company size categories to just 4 categories and One-Hot Encoding applied to this feature "
    )
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("company_size distribusi")
        sns.countplot(x="company_size", data=df)
        plt.xticks(rotation=60)
        st.pyplot(plt)  # Display the first plot
    df["company_size"] = df["company_size"].apply(
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
    with col2:
        plt.figure(figsize=(8, 6))
        plt.title("company_size new category")
        sns.countplot(x="company_size", data=df)
        plt.xticks(rotation=60)
        st.pyplot(plt)  # Display the first plot
    df = pd.get_dummies(df, columns=["company_size"], drop_first=False)
    st.dataframe(df.reset_index().sample(10).T.tail(4).T, use_container_width=True)
    st.write("---")
    st.text("company_type")
    st.write("One-Hot Encoding applied to this feature ")
    col1, col2 = st.columns(2)  # Create two columns

    # First plot: Before Label Encoding
    with col1:
        plt.figure(figsize=(8, 6))
        plt.title("company_type distribution")
        sns.countplot(x="company_type", data=df)
        plt.xticks(rotation=30)
        st.pyplot(plt)  # Display the first plot
    df = pd.get_dummies(df, columns=["company_type"], drop_first=False)
    with col2:
        st.dataframe(df.reset_index().sample(10).T.tail(7).T, use_container_width=True)


# Section: Feature Engineering
elif section == "Feature Engineering":
    df["last_new_job"] = df["last_new_job"].apply(
        lambda x: "Never" if x == "never" else x
    )  # just reads nicer
    df["enrolled_university"][
        df["enrolled_university"] == "no_enrollment"
    ] = "No Enrollment"  # just reads nicer
    df["company_size"] = df["company_size"].apply(
        lambda x: "10-49" if x == "10/49" else x
    )  # diff replacement method

    df["experience"] = df["experience"].apply(lambda x: "0" if x == "<1" else x)
    df["experience"] = df["experience"].apply(lambda x: "21" if x == ">20" else x)

    df["company_size"].fillna("0", inplace=True)
    df["company_type"].fillna("Unknown", inplace=True)
    df["major_discipline"].fillna("Unknown", inplace=True)
    df["gender"].fillna("Other", inplace=True)
    df.dropna(inplace=True)
    df["training_hours"] = np.log1p(df["training_hours"])

    df = df[(zscore(df["training_hours"]) > -3) & (zscore(df["training_hours"]) < 3)]

    df["relevent_experience"] = [
        1 if i == "Has relevent experience" else 0 for i in df["relevent_experience"]
    ]

    # Label encoding
    df["education_level"] = [
        (
            5
            if i == "Phd"
            else (
                4
                if i == "Masters"
                else 3 if i == "Graduate" else 2 if i == "High School" else 1
            )
        )
        for i in df["education_level"]
    ]

    # Label encoding
    df["experience"] = df["experience"].astype(int)

    # Second plot: After Label Encoding

    # Label encoding
    df["last_new_job"] = [
        (
            5
            if i == ">4"
            else (
                4
                if i == "4"
                else 3 if i == "3" else 2 if i == "2" else 1 if i == "1" else 0
            )
        )
        for i in df["last_new_job"]
    ]

    df = pd.get_dummies(df, columns=["gender"], drop_first=False)

    df = pd.get_dummies(df, columns=["enrolled_university"], drop_first=False)

    df = pd.get_dummies(df, columns=["major_discipline"], drop_first=False)

    df["company_size"] = df["company_size"].apply(
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

    df = pd.get_dummies(df, columns=["company_size"], drop_first=False)

    df = pd.get_dummies(df, columns=["company_type"], drop_first=False)

    st.header("Feature Engineering")
    st.write("---")
    st.subheader("Feature Selection")
    st.write(f"Currently we have {df.shape[1]-1} feature and 1 target")
    st.write(
        "We found that enrollee_id is merely a unique identifier for each employee, and the city feature has too many unique values but is already represented by the city_development_index. Therefore, we will drop these two columns."
    )
    df = df.drop(["enrollee_id", "city"], axis=1)
    st.dataframe(df.sample(10), use_container_width=True)

    st.subheader("Additional Feature")
    st.write(
        "Experience is often associated with the training attended. Someone with more experience may tend to participate in fewer training sessions due to already mastered skills, or conversely, someone with less experience may attend more training sessions. We can create a new feature that combines training_hours and experience by calculating the ratio of training hours per year of experience."
    )
    df["training_per_experience"] = np.expm1(df["training_hours"]) / (
        df["experience"] + 1
    )
    # Urutkan feature
    columns_order = [
        "city_development_index",
        "relevent_experience",
        "education_level",
        "experience",
        "last_new_job",
        "training_hours",
        "gender_Female",
        "gender_Male",
        "gender_Other",
        "enrolled_university_Full time course",
        "enrolled_university_No Enrollment",
        "enrolled_university_Part time course",
        "major_discipline_Arts",
        "major_discipline_Business Degree",
        "major_discipline_Humanities",
        "major_discipline_No Major",
        "major_discipline_Other",
        "major_discipline_STEM",
        "major_discipline_Unknown",
        "company_size_Large",
        "company_size_Medium",
        "company_size_Small",
        "company_size_Unknown",
        "company_type_Early Stage Startup",
        "company_type_Funded Startup",
        "company_type_NGO",
        "company_type_Other",
        "company_type_Public Sector",
        "company_type_Pvt Ltd",
        "company_type_Unknown",
        "training_per_experience",
        "target",
    ]
    df = df[columns_order]
    st.write(f"Now we have {df.shape[1]-1} feature and 1 Target")

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        fmt=".2f",
        annot_kws={"size": 12, "rotation": 45},
    )

    # Menambahkan judul
    plt.title("Heatmap of Correlation Matrix")
    st.pyplot(plt)

# Section: Split Data
if section == "Split Data Train-Test":
    st.header("Split Data Tain-Test")
    st.write(
        f"We have {df2.shape[0]} rows of data and will split this into 80% training data  and 20% testing data"
    )
    X = df2.drop(columns=["target"])
    y = df2["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    # Create a pie chart for train and test data proportions
    labels = ["Training", "Testing"]
    sizes = [len(X_train), len(X_test)]
    colors = ["#66b3ff", "#99ff99"]
    explode = (0.1, 0)  # Slightly explode the 'Pelatihan' section for visual effect

    plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
    )
    plt.axis("equal")  # Ensure the chart is circular
    plt.title("Train-Test Proportion")
    st.pyplot(plt)

# Section: Dataset Overview
if section == "Imbalance Data":

    X = df2.drop(columns=["target"])
    y = df2["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    st.header("Imbalance Data")
    st.write(
        "We use the oversampling method called SMOTE for imbalanced data and apply this method only to the training data."
    )
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # Menghitung jumlah data sebelum dan sesudah SMOTE
    sizes_before = [y_train.value_counts().get(0, 0), y_train.value_counts().get(1, 0)]
    sizes_after = [
        y_train_resampled.value_counts().get(0, 0),
        y_train_resampled.value_counts().get(1, 0),
    ]

    # Plot untuk sebelum SMOTE
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # Menampilkan pie chart sebelum SMOTE
    plt.pie(
        sizes_before,
        labels=[
            f"Class 0: {sizes_before[0]} ({sizes_before[0] / sum(sizes_before) * 100:.1f}%)",
            f"Class 1: {sizes_before[1]} ({sizes_before[1] / sum(sizes_before) * 100:.1f}%)",
        ],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff"],
        startangle=140,
        shadow=True,
    )
    plt.title("Before SMOTE")

    # Plot untuk sesudah SMOTE
    plt.subplot(1, 2, 2)  # Menampilkan pie chart setelah SMOTE
    plt.pie(
        sizes_after,
        labels=[
            f"Class 0: {sizes_after[0]} ({sizes_after[0] / sum(sizes_after) * 100:.1f}%)",
            f"Class 1: {sizes_after[1]} ({sizes_after[1] / sum(sizes_after) * 100:.1f}%)",
        ],
        autopct="%1.1f%%",
        colors=["#ff9999", "#66b3ff"],
        startangle=140,
        shadow=True,
    )
    plt.title("After SMOTE")

    plt.tight_layout()
    st.pyplot(plt)


# Section: Scaling
if section == "Scaling Data":

    X = df2.drop(columns=["target"])
    y = df2["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    st.header("Scaling data")
    st.write(
        "We use StandarScaler to scale all of the features and applied this scaler to X_train_resampled(data after SMOTE) and X_test"
    )
    scaler = StandardScaler()
    scaler.fit(X_train_resampled)
    X_train_scaled = scaler.transform(X_train_resampled)  # Fit dan transform data latih
    X_test_scaled = scaler.transform(X_test)  # Hanya transform data uji
    # Menampilkan grafik distribusi fitur sebelum dan sesudah scaling
    plt.figure(figsize=(12, 6))

    # Plot distribusi data sebelum scaling (X_train_resampled)
    plt.subplot(1, 2, 1)
    sns.boxplot(data=X_train_resampled)
    plt.title("Data Distribution Before Scaling")
    # Plot distribusi data setelah scaling (X_train_scaled)
    plt.subplot(1, 2, 2)
    sns.boxplot(data=X_train_scaled)
    plt.title("Data Distribution After Scaling")

    plt.tight_layout()
    st.pyplot(plt)
