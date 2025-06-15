import streamlit as st
import requests

# ✅ Correct API URL (single slash)
API_URL = "http://127.0.0.1:8000/predict"

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

# Submit button
if st.button("Predict Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        # Make API request
        response = requests.post(API_URL, json=input_data)

        # Show raw response for debugging
        st.write("Status Code:", response.status_code)
        st.write("Raw Response Text:", response.text)

        # Try to parse JSON
        result = response.json()

        # If prediction is successful
        if "predictted_category" in result:
            st.success(f"✅ Predicted Insurance Premium Category: **{result['predictted_category']}**")
        else:
            st.warning("⚠️ Unexpected response format.")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the FastAPI backend. Make sure it's running.")
    except Exception as e:
        st.error("❌ Failed to parse JSON from the response.")
        st.write("Error details:", str(e))
