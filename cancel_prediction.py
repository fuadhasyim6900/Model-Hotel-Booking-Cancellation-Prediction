import streamlit as st
import pandas as pd
import os
from joblib import load

# Load trained Decision Tree pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "DT_Tuned_pipeline.joblib")


@st.cache_resource
def load_model():
    return load(MODEL_PATH)


model = load_model()
st.title("Hotel Booking Cancellation Predictor")

# Sidebar inputs
st.sidebar.header("Booking Details")

# Categorical
deposit_type = st.sidebar.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
customer_type = st.sidebar.selectbox("Customer Type", ['Transient', 'Transient-Party', 'Contract', 'Group'])
reserved_room_type = st.sidebar.selectbox("Reserved Room Type", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'])
market_segment = st.sidebar.selectbox("Market Segment", ['Online TA', 'Direct', 'Corporate', 'Groups', 'Complementary', 'Aviation', 'Undefined', 'Offline TA/TO'])
country = st.sidebar.selectbox("Country", ['Portugal', 'Non Portugal', 'Other'])
meal = st.sidebar.selectbox("Meal", ['BB', 'HB', 'FB', 'SC', 'Undefined'])
distribution_channel = st.sidebar.selectbox("Distribution Channel", ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'])

# Numerical
lead_time = st.sidebar.number_input("Lead Time (days)", min_value=0, max_value=600, value=30, step=1)
adr = st.sidebar.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
total_of_special_requests = st.sidebar.number_input("Total Special Requests", min_value=0, max_value=5, value=0, step=1)
arrival_date_day_of_month = st.sidebar.number_input("Arrival Day of Month", min_value=1, max_value=31, value=1, step=1)
arrival_date_week_number = st.sidebar.number_input("Arrival Week Number", min_value=1, max_value=53, value=1, step=1)
stays_in_week_nights = st.sidebar.number_input("Stays in Week Nights", min_value=0, max_value=30, value=1, step=1)
stays_in_weekend_nights = st.sidebar.number_input("Stays in Weekend Nights", min_value=0, max_value=14, value=0, step=1)
required_car_parking_spaces = st.sidebar.number_input("Required Car Parking Spaces", min_value=0, max_value=5, value=0, step=1)
adults = st.sidebar.number_input("Adults", min_value=0, max_value=10, value=2, step=1)
children = st.sidebar.number_input("Children", min_value=0, max_value=5, value=0, step=1)
babies = st.sidebar.number_input("Babies", min_value=0, max_value=5, value=0, step=1)
previous_cancellations = st.sidebar.number_input("Previous Cancellations", min_value=0, max_value=10, value=0, step=1)
previous_bookings_not_canceled = st.sidebar.number_input("Previous Bookings Not Canceled", min_value=0, max_value=20, value=0, step=1)
is_repeated_guest = st.sidebar.selectbox("Repeated Guest?", [0, 1])

# Build input DataFrame
total_guest = adults + children + babies
total_night = stays_in_week_nights + stays_in_weekend_nights

input_data = pd.DataFrame({
    'lead_time': [lead_time],
    'deposit_type': [deposit_type],
    'adr': [adr],
    'total_of_special_requests': [total_of_special_requests],
    'arrival_date_day_of_month': [arrival_date_day_of_month],
    'arrival_date_week_number': [arrival_date_week_number],
    'stays_in_week_nights': [stays_in_week_nights],
    'stays_in_weekend_nights': [stays_in_weekend_nights],
    'required_car_parking_spaces': [required_car_parking_spaces],
    'adults': [adults],
    'children': [children],
    'babies': [babies],
    'total_guest': [total_guest],
    'total_night': [total_night],
    'customer_type': [customer_type],
    'reserved_room_type': [reserved_room_type],
    'market_segment': [market_segment],
    'country': [country],
    'meal': [meal],
    'distribution_channel': [distribution_channel],
    'previous_cancellations': [previous_cancellations],
    'previous_bookings_not_canceled': [previous_bookings_not_canceled],
    'is_repeated_guest': [is_repeated_guest],
    'agent': [0],  
    'company': [0],
    'booking_changes': [0],           
    'arrival_date_month': ['January']  
})

# Prediction
if st.button("Predict Cancellation"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ This booking is likely to be CANCELED\n\n"
            f"Cancellation Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ This booking is likely to be HONORED\n\n"
            f"Cancellation Probability: {probability:.2%}"
        )


