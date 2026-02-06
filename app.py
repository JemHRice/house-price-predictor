import streamlit as st
import pandas as pd
import pickle

st.title(" House Price Predictor")
st.write(
    """A machine learning model trained on real housing data.
         Enter your housing data to get real estimates!"""
)


# Load pre-trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# Get feature names
feature_names = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Air Conditioning",
    "Parking",
    "Preferred Area",
    "Fully Furnished",
    "Semi Furnished",
    "Unfurnished",
]

# Sidebar inputs
st.sidebar.header("House Features")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 6, 2)
area = st.sidebar.slider("Area (sq ft)", 500, 33000, 5000)
air_conditioning = st.sidebar.checkbox("Air Conditioning", value=True)
parking = st.sidebar.slider("Parking Spaces", 0, 3, 1)
preferred_area = st.sidebar.checkbox("Preferred Area", value=False)
selected_furnishing = st.sidebar.selectbox(
    "Furnishing Status", ["Fully Furnished", "Semi Furnished", "Unfurnished"]
)

# Build input dataframe with all features in correct order
user_input = {
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Air Conditioning": int(air_conditioning),
    "Parking": parking,
    "Preferred Area": int(preferred_area),
    "Fully Furnished": 1 if selected_furnishing == "Fully Furnished" else 0,
    "Semi Furnished": 1 if selected_furnishing == "Semi Furnished" else 0,
    "Unfurnished": 1 if selected_furnishing == "Unfurnished" else 0,
}

user_input_df = pd.DataFrame([user_input])

# Make prediction
prediction = model.predict(user_input_df)[0]

# Display results
st.subheader("Predicted Price")
st.metric("Estimated Value", f"${prediction:,.0f}")

st.subheader("Your Input Features")
# Create display dataframe for user
display_input = {
    "Area (sqft)": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Air Conditioning": "Yes" if air_conditioning else "No",
    "Parking": parking,
    "Preferred Area": "Yes" if preferred_area else "No",
    "Furnished?": selected_furnishing.replace(" Furnished", "").replace(
        "Unfurnished", "None"
    ),
}

st.write(pd.DataFrame([display_input]))

st.subheader("Model Information")
st.write(f"**Model:** Linear Regression with {len(feature_names)} features")
st.write(f"**Expected Error:** ±$850,000 on average *(expect improvements soon!)*")
st.write(
    f"**Source:** [Kaggle Housing Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)"
)
