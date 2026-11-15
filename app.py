
import joblib
import pandas as pd
import streamlit as st
from datetime import date, time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bus Delay Prediction",
    page_icon="🚌",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #f1f5f9;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #12345a;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    color: #12345a;
    font-size: 21px;
    font-weight: bold;
    padding-bottom: 10px;
}

div.stButton > button {
    width: 100%;
    background-color: #1261a0;
    color: white;
    height: 48px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #0b4778;
    color: white;
}

.result-card {
    background-color: #dbeafe;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border-left: 6px solid #1261a0;
}

.result-number {
    font-size: 38px;
    font-weight: bold;
    color: #12345a;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    day_encoder = joblib.load("E:\\Bus Delay Prediction ML Project\\pkl files\\label_day.pkl")
    scaler = joblib.load("E:\\Bus Delay Prediction ML Project\\pkl files\\scaled.pkl")
    model = joblib.load("E:\\Bus Delay Prediction ML Project\\pkl files\\randomforestregressor.pkl")
    return day_encoder, scaler, model


day_encoder, scaler, model = load_model()


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🚌 BUS DELAY PREDICTION SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Machine Learning Based Bus Delay Estimation</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- INPUT SECTION ----------------
st.markdown(
    '<div class="section-title">🚍 Bus Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    route = st.number_input(
        "Route Number",
        min_value=0,
        step=1
    )

    vehicle = st.number_input(
        "Vehicle Number",
        min_value=0,
        step=1
    )

    day_value = st.date_input(
        "Travel Date",
        value=date.today()
    )

with col2:
    time_value = st.time_input(
        "Travel Time",
        value=time(12, 0)
    )

    direction = st.selectbox(
        "Direction",
        ["North", "South", "East", "West"]
    )

    min_gap = st.number_input(
        "Minimum Gap (Minutes)",
        min_value=0,
        step=1
    )


st.divider()


# ---------------- INCIDENT SECTION ----------------
st.markdown(
    '<div class="section-title">⚠️ Incident Information</div>',
    unsafe_allow_html=True
)

incident = st.selectbox(
    "Select Incident Type",
    [
        "Mechanical",
        "Operations",
        "Security",
        "Collision",
        "Diversion",
        "General Delay",
        "Emergency Services",
        "Utilized Off Route",
        "Cleaning",
        "Investigation",
        "Vision",
        "Road Blocked",
        "Tire Puncher",
    ]
)


# ---------------- PREDICTION ----------------
incident_classes = [
    "Cleaning",
    "Collision",
    "Diversion",
    "Emergency Services",
    "General Delay",
    "Investigation",
    "Mechanical",
    "Operations",
    "Road Blocked",
    "Security",
    "Tire Puncher",
    "Utilized Off Route",
    "Vision",
]

direction_codes = {
    "East": 0,
    "North": 1,
    "South": 2,
    "West": 3
}


st.divider()

left, right = st.columns([3, 1])

with left:
    st.caption("Check the entered information before predicting the delay.")

with right:
    predict_button = st.button("🔍 Predict Delay", use_container_width=True)


if predict_button:

    features = pd.DataFrame(
        [[
            route,
            day_encoder.transform(
                [day_value.strftime("%A")]
            )[0],
            incident_classes.index(incident),
            min_gap,
            direction_codes[direction],
            vehicle,
            day_value.day,
            day_value.month,
            day_value.year,
            time_value.hour,
            time_value.minute
        ]],
        columns=[
            "Route",
            "Day",
            "Incident",
            "Min Gap",
            "Direction",
            "Vehicle",
            "Day_Date",
            "Month",
            "Year",
            "Hour",
            "Minute"
        ]
    )

    prediction = model.predict(
        scaler.transform(features)
    )[0]

    prediction = max(0, round(prediction))

    st.markdown("---")

    st.markdown(
        '<div class="result-card">'
        '<h3 style="color:#12345a;">Predicted Bus Delay</h3>'
        f'<div class="result-number">{prediction} Minutes</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Estimated delay for Route No {route}, "
        f"Vehicle {vehicle}."
    )


# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">'
    'Bus Delay Prediction System | Powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
