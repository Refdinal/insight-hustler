import streamlit as st

st.set_page_config(
    page_title="Model Prediction",
    page_icon="📈",
)

st.title("Prediction Results")
st.markdown(
    """
📈 **Predict whether an employee will leave the company.**
"""
)

# Input form
st.subheader("Enter Employee Details")
age = st.slider("Age", min_value=18, max_value=65, value=30)
job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
monthly_income = st.number_input(
    "Monthly Income", min_value=1000, max_value=20000, value=5000
)

# Prediction button
if st.button("Predict"):
    # Mocked prediction logic
    prediction = "Leave" if job_satisfaction < 3 else "Stay"
    st.subheader("Prediction Result:")
    st.success(f"The employee is likely to **{prediction}**.")
