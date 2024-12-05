import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        "Descriptive Statistics",
        "Univariate Analysis",
        "Multivariate Analysis",
        "Business Insights",
    ],
)


# Load dummy data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Refdinal/insight-hustler/refs/heads/master/data/aug_train.csv"
    )
    return df


df = load_data()
nums = ["enrollee_id", "city_development_index", "training_hours", "target"]
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

# Section: Descriptive Statistics
if section == "Descriptive Statistics":

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write("Data Type and Columns:")

    # Numerical Data
    st.write("Numerical Data")
    st.dataframe(df[nums].describe().T, use_container_width=True)

    # Categorical Data
    st.write("Categorical Data")
    st.dataframe(df[cats].describe().T, use_container_width=True)

    # Loop through categorical columns
    for col in cats:
        st.write("----")
        st.write(f"""**Value Counts for Column: {col}**""")

        # Calculate full value counts and display them
        value_counts = df[col].value_counts()
        st.dataframe(value_counts, use_container_width=True)

        # Display the missing values count for the column
        missing_values = df[col].isnull().sum()
        st.write(f"Missing Values Count for '{col}': {missing_values}")

        # Get top 10 for plotting
        top = value_counts.head(10)

        # Plot top 10 value counts
        fig, ax = plt.subplots()
        sns.barplot(x=top.index, y=top.values, ax=ax, palette="viridis")
        ax.set_title(f"Top Value Counts for {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels by 45 degrees

        # Add data labels on top of bars
        for i in ax.containers:
            ax.bar_label(i)

        # Display the plot
        st.pyplot(fig)


# # Section: Univariate Analysis
# elif section == "Univariate Analysis":
#     st.subheader("Univariate Analysis")
#     st.write("Analyzing individual features to identify patterns.")

#     st.write("### Distribution of Age")
#     fig, ax = plt.subplots()
#     sns.histplot(data["Age"], kde=True, bins=10, ax=ax, color="blue")
#     st.pyplot(fig)

#     st.write("### Attrition Rate")
#     attrition_count = data["Attrition"].value_counts()
#     st.bar_chart(attrition_count)

# # Section: Multivariate Analysis
# elif section == "Multivariate Analysis":
#     st.subheader("Multivariate Analysis")
#     st.write("Exploring relationships between different features.")

#     st.write("### Correlation Matrix")
#     corr = data.select_dtypes(include=["int64", "float64"]).corr()
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

#     st.write("### Monthly Income vs. Job Satisfaction")
#     fig, ax = plt.subplots()
#     sns.boxplot(x="JobSatisfaction", y="MonthlyIncome", data=data, ax=ax)
#     st.pyplot(fig)

# # Section: Business Insights
# elif section == "Business Insights":
#     st.subheader("Business Insights")
#     st.write("Key findings from the data:")

#     st.markdown(
#         """
#     - **Attrition Rate**: Employees with lower job satisfaction tend to leave more frequently.
#     - **Education and Income**: Higher education levels generally correlate with higher income.
#     - **Age and Attrition**: Younger employees are more likely to leave the company.
#     """
#     )

#     st.write("### Attrition Rate by Education")
#     attrition_edu = (
#         data.groupby("Education")["Attrition"]
#         .value_counts(normalize=True)
#         .unstack()
#         .fillna(0)
#     )
#     st.bar_chart(attrition_edu)
