import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("best_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Employee Salary Prediction", page_icon="🪙", layout="centered"
)
st.title("🪙 Employee Salary Prediction App")
st.markdown("Predict if an employee earns >50K or <=50K")

# --- Detect numeric vs categorical columns ---
numeric_cols = []
categorical_cols = []

for col in feature_cols:
    # numeric columns remain as-is
    if col in ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]:
        numeric_cols.append(col)
    else:
        # all OHE columns are categorical
        base_name = "_".join(col.split("_")[:-1])  # get original category base
        if base_name not in categorical_cols:
            categorical_cols.append(base_name)

# --- Build input widgets dynamically ---
numeric_inputs = {}
numeric_inputs['age'] = st.sidebar.slider("Age",18,65,25)
numeric_inputs['hours-per-week'] = st.sidebar.slider("Hours per week",1,80,40)
numeric_inputs['experience'] = st.sidebar.slider("Years of Experience",0,40,5)
for col in numeric_cols:
    # You can adjust min/max defaults as needed
    if col not in ['age','hours-per-week','experience']:
        numeric_inputs[col] = st.sidebar.slider(col, 0, 100000, 0)

categorical_inputs = {}
for cat in categorical_cols:
    # collect all columns starting with this category
    options = [c[len(cat) + 1 :] for c in feature_cols if c.startswith(cat + "_")]
    categorical_inputs[cat] = st.sidebar.selectbox(cat, options)

# --- Build one-hot encoded DataFrame ---
input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

# Set numeric values
for col, val in numeric_inputs.items():
    input_df[col] = val

# Set categorical OHE values
for cat, val in categorical_inputs.items():
    col_name = f"{cat}_{val}"
    if col_name in input_df.columns:
        input_df[col_name] = 1

st.write("### Input Data")
st.write(input_df)

# --- Predict ---
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"Prediction: {'>50K' if prediction[0]==1 else '<=50K'}")
