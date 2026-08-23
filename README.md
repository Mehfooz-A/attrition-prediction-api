
# Employee Attrition Prediction — API + Dashboard

I built this to predict which employees are likely to quit, based on their HR data — and to make that prediction usable outside of a Jupyter notebook. It's a logistic regression model, served through a real API, with a simple dashboard on top so someone in HR can actually use it without touching any code.

## What it does

- Give it an employee's details, get back a Yes/No attrition risk plus a probability

- Trained on the IBM HR Analytics dataset — 1,470 real (anonymized) employee records

- The prediction logic runs behind a proper API, so it's not locked to this one dashboard — anything could plug into it

- The dashboard itself is plain-English: dropdowns, sliders, a button, and a clear result. No JSON in sight.

## Why I built it this way

I originally did an attrition analysis for a stats course, in R. It worked, but it was static — a report, not a tool. This is that same idea rebuilt properly: retrained in Python from scratch (R and Python encode categorical data differently, so porting the old code wasn't worth it), then wired into an actual API and a dashboard so it's something a real HR person could open and use.

## Tech stack

- scikit-learn — the model itself, plus feature scaling

- FastAPI — serves predictions over a real endpoint

- Streamlit — the dashboard that talks to the API

- pandas / joblib — data prep and saving the trained model

## How it works, roughly

1. Employee data gets encoded (categories into numbers) and scaled

2. A logistic regression model predicts attrition risk, trained with class weighting since only about 16% of employees in this dataset actually left

3. That model is served through a FastAPI /predict endpoint

4. The Streamlit dashboard collects the same info through a form, sends it to the API, and shows the result as a plain colored message instead of a raw score

## How well it actually performs

On a held-out test set (20% of the data, never seen during training):

| | Precision | Recall | F1 |

|---|---|---|---|

| Stayed | 0.90 | 0.91 | 0.90 |

| Left | 0.49 | 0.45 | 0.47 |

Worth being upfront about: this isn't a "99% accurate" headline number, and that's intentional. The model is tuned to catch more real attrition cases (higher recall) at the cost of some false alarms, since missing someone who's actually about to leave felt like the more expensive mistake for this problem. It's a dial you can turn, not a fixed answer, and I go into that trade-off in more depth in ARCHITECTURE.md.

## What actually drives attrition, according to this model

Overtime is the single biggest factor. Frequent travel and a few specific roles (Lab Technician, in particular) also stand out. On the flip side, more tenure and higher satisfaction scores both pull risk down. None of that's shocking, honestly, but having it come directly out of the model's own coefficients, rather than a gut feeling, is the actual value here.

## Run it yourself

Terminal 1, the API:

```bash

git clone https://github.com/Mehfooz-A/attrition-prediction-api.git

cd attrition-prediction-api

pip install -r requirements.txt

python -m uvicorn main:app --reload

```

Terminal 2, the dashboard:

```bash

python -m streamlit run dashboard.py

```

Locally, the API runs at http://127.0.0.1:8000 and the dashboard at http://localhost:8501. If you're running the dashboard locally against your own local API, update API_URL near the top of dashboard.py back to the localhost address.


API's at http://127.0.0.1:8000 (with interactive docs at /docs), dashboard's at http://localhost:8501.



[ARCHITECTURE.md](./ARCHITECTURE.md) covers the actual build process — what went wrong along the way, why I picked logistic regression over something fancier, and how I landed on the final model settings.



This is Module 2 of a bigger project I'm building — an HR toolkit that also includes a document Q&A chatbot (Module 1) and, eventually, explainability and reporting on top of this same model.

