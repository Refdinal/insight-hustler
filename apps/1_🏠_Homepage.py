import streamlit as st

st.set_page_config(
    page_title="Homepage",
    page_icon="🏠",
)

st.title("Employee Attrition Prediction Dashboard")
st.markdown(
    """
**Welcome to Insight Hustler's Project!** 👋

This application is designed to analyze employee attrition rates and predict the likelihood of an employee leaving their job. By understanding the key factors influencing this decision, companies can take proactive measures to retain valuable employees.

---

### **Problem Statement**
How can we identify the factors influencing an employee's decision to leave their job and predict their likelihood of doing so? 

---

### **Project Goals**
1. Build a predictive model to assess the probability of employee attrition.
2. Interpret the most critical factors influencing attrition.
3. Provide actionable insights for HR to develop effective retention strategies.

---

### **Team Members**
- **Mentor**: Salsabila Nur Yasmin  
- **Data Enginner & Data Scientist** : Annisa Fatihaturrahma  
- **Project Manager & Data Analyst** : Refdinal F  

---

### **Objectives**
- Create a robust predictive model using features like education, work experience, etc.
- Provide insights into employee attrition trends and actionable recommendations for HR.

---

Navigate through the sidebar to explore the data, preprocessing steps, modeling, and prediction results. 🚀  

---

### **Contribute**
Fork our repository on [GitHub](https://github.com/Refdinal/insight-hustler) and contribute to our project!
"""
)
