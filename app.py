import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
st.set_page_config(
    page_title="Porter Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)
@st.cache_resource
def load_files():

    model = tf.keras.models.load_model(
        "porter_delivery_model.keras"
    )

    preprocessor = joblib.load(
        "porter_preprocessor.pkl"
    )

    store_frequency = joblib.load(
        "porter_store_frequency.pkl"
    )

    feature_info = joblib.load(
        "porter_feature_info.pkl"
    )

    return model, preprocessor, store_frequency, feature_info


model, preprocessor, store_frequency, feature_info = load_files()
st.title("🚚 Porter Delivery Time Predictor")

st.write(
    "AI-powered delivery time prediction using a "
    "Neural Network regression model."
)

st.info(
    "Enter the order details below to estimate "
    "the delivery time in minutes."
)
st.header("📦 Order Details")
market_id = st.number_input(
    "Market ID",
    min_value=0,
    value=1,
    step=1
)
store_id = st.text_input(
    "Store ID",
    placeholder="Enter Store ID"
)
store_primary_category = st.text_input(
    "Store Primary Category",
    value="American"
)
order_protocol = st.number_input(
    "Order Protocol",
    min_value=0,
    value=1,
    step=1
)
total_items = st.number_input(
    "Total Items",
    min_value=1,
    value=2,
    step=1
)
num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=1,
    value=2,
    step=1
)
min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0.0,
    value=100.0,
    step=1.0
)
max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0.0,
    value=300.0,
    step=1.0
)
total_onshift_partners = st.number_input(
    "Total On-Shift Partners",
    min_value=0,
    value=10,
    step=1
)
total_busy_partners = st.number_input(
    "Total Busy Partners",
    min_value=0,
    value=5,
    step=1
)
total_outstanding_orders = st.number_input(
    "Total Outstanding Orders",
    min_value=0,
    value=10,
    step=1
)
st.header("🕐 Order Time Information")
order_hour = st.slider(
    "Order Hour",
    min_value=0,
    max_value=23,
    value=12
)
order_day = st.slider(
    "Order Day",
    min_value=1,
    max_value=31,
    value=15
)
order_month = st.slider(
    "Order Month",
    min_value=1,
    max_value=12,
    value=6
)
day_of_week = st.slider(
    "Day of Week",
    min_value=0,
    max_value=6,
    value=2,
    help="0 = Monday, 1 = Tuesday, ..., 6 = Sunday"
)
is_weekend = st.selectbox(
    "Is Weekend?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
st.divider()
if st.button(
    "🚀 Predict Delivery Time",
    use_container_width=True
):
    if store_id.strip() == "":
        st.warning("Please enter a Store ID.")
        st.stop()
    store_id_clean = store_id.strip()

    store_freq = store_frequency.get(
        store_id_clean,
        0
    )
    input_data = pd.DataFrame({

        "market_id": [market_id],

        "store_primary_category": [
            store_primary_category
        ],

        "order_protocol": [
            order_protocol
        ],

        "total_items": [
            total_items
        ],

        "num_distinct_items": [
            num_distinct_items
        ],

        "min_item_price": [
            min_item_price
        ],

        "max_item_price": [
            max_item_price
        ],

        "total_onshift_partners": [
            total_onshift_partners
        ],

        "total_busy_partners": [
            total_busy_partners
        ],

        "total_outstanding_orders": [
            total_outstanding_orders
        ],

        "order_hour": [
            order_hour
        ],

        "order_day": [
            order_day
        ],

        "order_month": [
            order_month
        ],

        "day_of_week": [
            day_of_week
        ],

        "is_weekend": [
            is_weekend
        ],

        "store_frequency": [
            store_freq
        ]
    })
    input_processed = preprocessor.transform(
        input_data
    )

    input_processed = np.asarray(
        input_processed,
        dtype="float32"
    )
    prediction = model.predict(
        input_processed,
        verbose=0
    )[0][0]


    prediction = max(
        0,
        float(prediction)
    )
    st.success(
        f"Estimated Delivery Time: "
        f"{prediction:.1f} minutes"
    )


    st.metric(
        label="Predicted Delivery Time",
        value=f"{prediction:.1f} minutes"
    )
    if prediction <= 30:

        st.write(
            "🟢 The order is expected to be delivered quickly."
        )

    elif prediction <= 60:

        st.write(
            "🟡 The order has an expected moderate delivery time."
        )

    else:

        st.write(
            "🔴 The order may require a longer delivery time."
        )