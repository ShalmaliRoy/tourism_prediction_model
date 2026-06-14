import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Shalmali85/TourismPredictionModel", filename="tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Conversion Prediction App")
st.write("Enter the customer details below to predict whether they will purchase a travel package.")

col1, col2 = st.columns(2)

with col1:
    age                     = st.slider("Age", 18, 80, 35)
    type_of_contact         = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier               = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch       = st.slider("Duration of Pitch (min)", 1, 30, 10)
    occupation              = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Large Business", "Small Business"])
    gender                  = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_vis    = st.slider("Number of Persons Visiting", 1, 5, 2)
    number_of_followups     = st.slider("Number of Follow-ups", 1, 6, 3)

with col2:
    product_pitched         = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Stars", [3, 4, 5])
    marital_status          = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    number_of_trips         = st.slider("Number of Trips", 1, 10, 2)
    passport                = st.selectbox("Passport", [0, 1])
    pitch_satisfaction      = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car                 = st.selectbox("Own Car", [0, 1])
    num_children_visiting   = st.slider("Number of Children Visiting", 0, 3, 0)
    designation             = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
    monthly_income          = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

# ── Build raw input dataframe (categorical as strings) ─────────────────────
raw_input = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_vis,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# ── One-hot encode and align with model's expected columns ─────────────────
input_encoded = pd.get_dummies(raw_input)

# Add any missing columns (set to 0) and drop any extra columns,
# then reorder to match exactly what the model was trained on
input_data = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# ── Predict ───────────────────────────────────────────────────────────────
if st.button("Predict"):
    prediction   = model.predict(input_data)[0]
    probability  = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to BUY a package. (Probability: {probability:.2%})")
    else:
        st.warning(f"Customer is unlikely to buy a package. (Probability: {probability:.2%})")
