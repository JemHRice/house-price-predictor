import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Sydney House Price Predictor 2026", layout="wide")

st.title("🏠 Sydney House Price Predictor 2026")
st.write(
    """
    Advanced AI model trained on 10,000+ Sydney property sales.
    Estimates 2026 prices with **±8.7% average accuracy**.
    Use the sidebar to enter property details. \n
    Real world prices WILL differ - Sydney's real estate market 
    is too volatile for 100% accuracy. Use this as a tool for 
    rough estimates.
"""
)


# Load model and data
@st.cache_resource
def load_model():
    """Load the advanced trained model"""
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_suburbs():
    """Load suburb list"""
    df = pd.read_csv("domain_properties_cleaned.csv")
    return sorted(df["suburb"].unique()), sorted(df["type"].unique())


try:
    model_package = load_model()
    suburbs, property_types = load_suburbs()

    # Sidebar inputs
    st.sidebar.header("📋 Property Details")

    # Suburb search
    suburb_search = st.sidebar.text_input(
        "🔍 Search Suburb", placeholder="Type suburb name..."
    )
    filtered_suburbs = (
        [s for s in suburbs if suburb_search.lower() in s.lower()]
        if suburb_search
        else suburbs[:10]
    )

    if suburb_search or len(suburbs) > 10:
        st.sidebar.write("**Matching suburbs:**")
        cols = st.sidebar.columns(2)
        suburb = None

        for idx, s in enumerate(filtered_suburbs[:20]):
            with cols[idx % 2]:
                if st.button(s, key=f"suburb_{idx}", use_container_width=True):
                    suburb = s
                    st.session_state.selected_suburb = s

        if "selected_suburb" in st.session_state:
            suburb = st.session_state.selected_suburb
        elif suburb is None:
            suburb = filtered_suburbs[0] if filtered_suburbs else suburbs[0]
    else:
        suburb = st.sidebar.selectbox("Or select from list:", suburbs)

    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 8, 2)
    parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)

    property_type = st.sidebar.selectbox("Property Type", property_types, index=0)

    property_size = st.sidebar.number_input(
        "Property Size (sqm)", min_value=50, max_value=2000, value=250
    )

    # Get model components
    model = model_package["model"]
    scaler = model_package["scaler"]
    label_encoders = model_package["label_encoders"]
    feature_columns = model_package["feature_columns"]
    numerical_cols = model_package["numerical_cols"]
    suburb_stats = model_package["suburb_stats"]
    kmeans = model_package["kmeans"]
    config = model_package["training_config"]

    # Prepare suburb info
    suburb_df = pd.DataFrame(suburb_stats)
    suburb_row = (
        suburb_df[suburb_df["suburb"] == suburb].iloc[0]
        if suburb in suburb_df["suburb"].values
        else None
    )

    # Create input dataframe with feature engineering
    user_input = pd.DataFrame(
        {
            "suburb": [suburb],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "parking": [parking],
            "property_size_sqm": [property_size],
            "type": [property_type],
            "property_inflation_index": [150.9],
            "price_cluster": [model_package["suburb_to_cluster"].get(suburb, 0)],
            "price_deviation_from_median": [0],  # Will predict this
            "size_ratio_to_suburb": [
                (
                    property_size / suburb_row["avg_size"]
                    if suburb_row is not None
                    else 1.0
                )
            ],
            "suburb_price_volatility": [
                (
                    suburb_row["price_std"] / suburb_row["median_price"]
                    if suburb_row is not None
                    else 0.3
                )
            ],
            "year_sold": [2026],
            "quarter_sold": [1],
            "rooms_total": [bedrooms + bathrooms],
            "parking_per_sqm": [parking / (property_size / 100)],
        }
    )

    # Encode categorical features
    user_input_encoded = user_input.copy()
    user_input_encoded["suburb"] = label_encoders["suburb"].transform([suburb])
    user_input_encoded["type"] = label_encoders["type"].transform([property_type])

    # Scale numerical features
    user_input_scaled = user_input_encoded.copy()
    user_input_scaled[numerical_cols] = scaler.transform(
        user_input_encoded[numerical_cols]
    )

    # Make prediction (price per sqm)
    price_per_sqm_pred = model.predict(user_input_scaled)[0]

    # Convert to absolute price
    predicted_price = price_per_sqm_pred * property_size

    # Calculate uncertainty range (based on MAPE of 8.7%)
    mape = config["mape"] / 100
    lower_bound = predicted_price * (1 - mape)
    upper_bound = predicted_price * (1 + mape)

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Estimated Price Range")
        st.metric("Predicted Price", f"${predicted_price:,.0f}")
        st.text(f"Likely Range: ${lower_bound:,.0f} — ${upper_bound:,.0f}")
        st.caption(f"±{config['mape']:.1f}% accuracy")

    with col2:
        st.subheader("📊 Suburb Context")
        if suburb_row is not None:
            st.metric("Suburb Median", f"${suburb_row['median_price']:,.0f}")
            cluster = int(suburb_row["price_cluster"])
            cluster_names = [
                "Luxury (3.5M+)",
                "Premium (2.7M-3.4M)",
                "Upper (2M-2.6M)",
                "Middle (1.4M-2M)",
                "Entry (600k-1.3M)",
            ]
            st.write(f"**Price Tier:** {cluster_names[cluster]}")
        else:
            st.write("*(Suburb data unavailable)*")

    # ============================================================
    # PROPERTY DETAILS
    # ============================================================

    st.subheader("📍 Your Property Details")

    details_df = pd.DataFrame(
        {
            "Feature": [
                "Suburb",
                "Type",
                "Bedrooms",
                "Bathrooms",
                "Parking",
                "Size (sqm)",
                "$/sqm (predicted)",
            ],
            "Value": [
                str(suburb),
                str(property_type),
                str(bedrooms),
                str(bathrooms),
                str(parking),
                str(property_size),
                f"${price_per_sqm_pred:,.0f}",
            ],
        }
    )

    st.dataframe(details_df, hide_index=True)

    # ============================================================
    # MODEL INFO
    # ============================================================

    st.subheader("📚 How Accurate Is This?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Error", f"±${config['mae_absolute']:,.0f}")
    with col2:
        st.metric("Accuracy (R²)", f"{config['r2_per_sqm']:.1%}")
    with col3:
        st.metric("Percentage Error", f"±{config['mape']:.1f}%")

    st.text(
        """
This prediction is based on:
- 10,373 Sydney property sales (2016-2021)
- Advanced XGBoost model with suburb clustering
- Feature engineering: 15 advanced indicators
- Price normalized by property size for accuracy
- 2026 inflation adjustment via RBA data

What this means:
- Useful for ballpark estimates and comparisons
- ±$146k average error on predictions around $1.5M
- Better for identifying over/undervalued properties
- Less useful for exact valuations (get a professional appraisal!)
    """
    )

except FileNotFoundError:
    st.error("❌ Model not found!")
    st.write("Run: `python train_model.py` to train the model first")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.write("Check that all files are present and model is trained correctly")

# Footer with dataset link
st.divider()
st.caption(
    "📊 [View Dataset on Kaggle](https://www.kaggle.com/datasets/alexlau203/sydney-house-prices)"
)
