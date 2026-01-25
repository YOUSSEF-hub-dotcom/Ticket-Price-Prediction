import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Flight Price AI Engine",
    page_icon="✈️",
    layout="wide"
)

AIRLINES = ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Multiple carriers", "GoAir", "Vistara", "Air Asia"]
CITIES = ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai", "Cochin", "Hyderabad"]

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .price-box { border: 2px solid #007bff; padding: 20px; border-radius: 10px; text-align: center; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("✈️ Advanced Flight Price Predictor")
st.info("This engine uses an XGBoost model deployed via MLflow to predict ticket prices.")

col_input, col_display = st.columns([1.5, 1])

with col_input:
    st.subheader("📍 Journey Details")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            airline = st.selectbox("Select Airline", AIRLINES)
            source = st.selectbox("Source City", CITIES)
            dep_date = st.date_input("Departure Date", datetime.now())
            dep_hour = st.slider("Departure Hour", 0, 23, 10)

        with col2:
            stops = st.number_input("Total Stops", 0, 4, 0)
            destination = st.selectbox("Destination City", CITIES)
            duration = st.number_input("Duration (Minutes)", 30, 3000, 120)
            arr_hour = st.slider("Expected Arrival Hour", 0, 23, 14)

        submit = st.form_submit_button("💰 Predict Ticket Price")

if submit:
    if source == destination:
        st.error("Source and Destination cannot be the same!")
    else:
        payload = {
            "Airline": airline,
            "Source": source,
            "Destination": destination,
            "Departure_Date": str(dep_date),
            "Dep_hour": dep_hour,
            "Arrival_hour": arr_hour,
            "Duration_mins": float(duration),
            "Total_Stops": stops
        }

        try:
            with st.spinner("🤖 AI is analyzing market trends..."):
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)

                if response.status_code == 200:
                    result = response.json()
                    with col_display:
                        st.markdown(f"""
                        <div class="price-box">
                            <h3>Predicted Ticket Price</h3>
                            <h1 style="color: #007bff;">₹ {result['predicted_price']}</h1>
                            <p style="color: gray;">Model Latency: {result['inference_time']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                elif response.status_code == 429:
                    st.error("Rate limit exceeded! Please wait a minute.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection Failed: {str(e)}")

st.markdown("---")
st.subheader("📜 Recent Search History")

try:
    history_res = requests.get("http://127.0.0.1:8000/history")
    if history_res.status_code == 200:
        hist_data = history_res.json()
        if hist_data:
            df = pd.DataFrame(hist_data)
            df.columns = ["ID", "Airline", "Source", "Destination", "Predicted Price", "Search Time"]
            st.dataframe(df.drop(columns=["ID"]), use_container_width=True)
        else:
            st.write("No history found.")
except:
    st.write("Could not load history.")