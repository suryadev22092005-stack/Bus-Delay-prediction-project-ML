
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


# ---------------- FEATURE COLUMNS ----------------

FEATURE_COLUMNS = [
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


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    base_path = r"E:\Bus Delay Prediction ML Project\pkl files"

    day_encoder = joblib.load(
        base_path + r"\label_day.pkl"
    )

    direc_encoder = joblib.load(
        base_path + r"\label_direction.pkl"
    )

    incident_encoder = joblib.load(
        base_path + r"\label_incident.pkl"
    )

    model = joblib.load(
        base_path + r"\randomforestregressor.pkl"
    )

    return day_encoder, direc_encoder, incident_encoder, model


day_encoder, direc_encoder, incident_encoder, model = load_model()


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🚌 BUS DELAY PREDICTION SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Machine Learning Based Bus Delay Estimation'
    '</div>',
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

    day_value = st.date_input(
        "Travel Date",
        value=date.today()
    )

    min_gap = st.number_input(
        "Minimum Gap (Minutes)",
        min_value=0,
        step=1
    )


with col2:

    direction = st.selectbox(
        "Direction",
        ["N", "S", "E", "W"]
    )

    vehicle = st.number_input(
        "Vehicle Number",
        min_value=0,
        step=1
    )

    time_value = st.time_input(
        "Travel Time",
        value=time(12, 0)
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
        "Tire Puncher"
    ]
)


# ---------------- PREDICTION BUTTON ----------------

st.divider()

left, right = st.columns([3, 1])

with left:

    st.caption(
        "Check the entered information before predicting the delay."
    )

with right:

    predict_button = st.button(
        "🔍 Predict Delay",
        use_container_width=True
    )


# ---------------- PREDICTION PROCESS ----------------

if predict_button:

    try:

        # STEP 1: Encode weekday

        day_name = day_value.strftime("%A")

        if day_name not in day_encoder.classes_:
            raise ValueError(
                "Selected weekday is not available in the encoder."
            )

        day_encoded = day_encoder.transform(
            [day_name]
        )[0]


        # STEP 2: Encode direction

        if direction not in direc_encoder.classes_:
            raise ValueError(
                "Selected direction is not available in the encoder."
            )

        direction_encoded = direc_encoder.transform(
            [direction]
        )[0]


        # STEP 3: Encode incident

        if incident not in incident_encoder.classes_:
            raise ValueError(
                "Selected incident is not available in the encoder."
            )

        incident_encoded = incident_encoder.transform(
            [incident]
        )[0]


        # STEP 4: Create input DataFrame

        features = pd.DataFrame(
            [[
                route,
                day_encoded,
                incident_encoded,
                min_gap,
                direction_encoded,
                vehicle,
                day_value.day,
                day_value.month,
                day_value.year,
                time_value.hour,
                time_value.minute
            ]],

            columns=FEATURE_COLUMNS
        )


        # STEP 5: Verify feature count

        if model.n_features_in_ != len(FEATURE_COLUMNS):

            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but received {len(FEATURE_COLUMNS)}."
            )


        # STEP 6: Verify feature names

        if hasattr(model, "feature_names_in_"):

            if list(model.feature_names_in_) != FEATURE_COLUMNS:

                raise ValueError(
                    "Feature names or feature order do not match "
                    "the training data."
                )


        # STEP 7: PREDICTION WITHOUT SCALING

        prediction = model.predict(features)[0]


        # STEP 8: Convert prediction into integer

        prediction = max(0, round(float(prediction)))


        # ---------------- DISPLAY RESULT ----------------

        st.markdown("---")

        st.markdown(
            '<div class="result-card">'
            '<h3 style="color:#12345a;">'
            'Predicted Bus Delay'
            '</h3>'
            f'<div class="result-number">'
            f'{prediction} Minutes'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            f"Estimated delay for Route No {route}, "
            f"Vehicle {vehicle}."
        )


    except ValueError as e:

        st.error(
            f"Prediction failed: {e}"
        )

    except Exception as e:

        st.error(
            f"An unexpected error occurred: {e}"
        )


# ---------------- FOOTER ----------------

st.markdown(
    '<div class="footer">'
    'Bus Delay Prediction System | Powered by Machine Learning'
    '</div>',
    unsafe_allow_html=True
)
