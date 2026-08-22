import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model, scaler, and expected column order — once, at startup
model = joblib.load("attrition_model.joblib")
scaler = joblib.load("scaler.joblib")
model_columns = joblib.load("model_columns.joblib")

app = FastAPI(title="Attrition Prediction API")

# Defines the exact shape of data this API expects to receive
class Employee(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    OverTime: str

@app.get("/")
def root():
    return {"message": "Attrition Prediction API is running"}

@app.post("/predict")
def predict(employee: Employee):
    # Convert the incoming request into a one-row DataFrame
    input_df = pd.DataFrame([employee.model_dump()])

    # One-hot encode the same way we did during training
    input_encoded = pd.get_dummies(input_df)

    # Add any missing columns (categories not present in this single row) as 0
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale using the SAME scaler fitted during training
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "attrition_risk": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability), 3)
    }
