import streamlit as st
import requests

st.set_page_config(page_title="Attrition Risk Checker", page_icon=":bar_chart:")
st.title("Employee Attrition Risk Checker")
st.caption("Enter an employee's details to check their predicted attrition risk.")

API_URL = "https://bigger-attrition-prediction-api-com.onrender.com/predict"

with st.form("employee_form"):
    st.subheader("Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, value=3)
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
        overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
        business_travel = st.selectbox(
            "Business Travel Frequency",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )

    with col2:
        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )
        job_role = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"]
        )
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree",
             "Human Resources", "Other"]
        )
        total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
        num_companies_worked = st.number_input("Number of Companies Worked At", min_value=0, value=1)

    submitted = st.form_submit_button("Check Attrition Risk")

if submitted:
    # Build the payload — every field the API's Employee model expects.
    # Fields not collected via the form get reasonable defaults.
    payload = {
        "Age": age,
        "DailyRate": 800,
        "DistanceFromHome": 5,
        "Education": 3,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": overtime,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        risk = result["attrition_risk"]
        probability = result["probability"]

        st.subheader("Result")

        if risk == "Yes":
            st.error(f"**High Attrition Risk** — {probability * 100:.1f}% predicted probability")
        else:
            st.success(f"**Low Attrition Risk** — {probability * 100:.1f}% predicted probability")

        st.progress(probability)

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the prediction API. Make sure it's running at http://127.0.0.1:8000")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
