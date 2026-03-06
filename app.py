import streamlit as st
import numpy as np
import pickle
from prediction import get_prediction
from config import *
from PIL import Image
import xgboost as xgb 


# ---------- LOAD ENCODERS ---------- #

def load_encoding():
    with open("model/checkpoint.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_encoding()

le_Day_of_week = data['le_Day_of_week']
le_Vehicle_driver_relation = data['le_Vehicle_driver_relation']
le_Road_surface_type = data['le_Road_surface_type']
le_Road_surface_conditions = data['le_Road_surface_conditions']
le_Type_of_collision = data['le_Type_of_collision']
le_Vehicle_movement = data['le_Vehicle_movement']
le_Work_of_casuality = data['le_Work_of_casuality']
le_Cause_of_accident = data['le_Cause_of_accident']


# ---------- LOAD MODEL ---------- #

model = xgb.Booster()
model.load_model("model/model_xgb.bin")


# ---------- PAGE CONFIG ---------- #

st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)


# ---------- OPTIONS ---------- #

options_time = ["Day", "Night"]

options_day = [
    "Sunday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday"
]

options_age = ["18-30", "31-50", "Over 51", "Under 18"]

options_vehicle_relation = ["Employee", "Owner"]

options_driver_exp = [
    "5-10yr", "2-5yr", "Above 10yr",
    "1-2yr", "Below 1yr", "No Licence"
]

options_vehicle_service = [
    "Below 1yr", "1-2yr", "2-5yrs",
    "5-10yrs", "Above 10yr"
]

options_road_surface = [
    "Asphalt roads", "Earth roads",
    "Asphalt roads with some distress",
    "Gravel roads", "other"
]

options_road_condition = [
    "Dry", "Wet or damp", "Snow",
    "Flood over 3cm. deep"
]

options_light = [
    "Darkness - no lighting",
    "Darkness - lights lit",
    "Daylight"
]

options_collision = [
    "Collision with roadside-parked vehicles",
    "Vehicle with vehicle collision",
    "Collision with roadside objects",
    "Collision with animals",
    "Rollover",
    "Fall from vehicles",
    "Collision with pedestrians",
    "With Train"
]

options_vehicle_move = [
    "Going straight", "U-Turn", "Moving Backward",
    "Turnover", "Waiting to go", "Getting off",
    "Reversing", "Parked", "Stopping",
    "Overtaking", "Entering a junction"
]

options_casualty_age = [
    "Under 18", "18-30", "31-50", "Over 51"
]

options_casualty_work = [
    "Driver", "Employee", "Self-employed",
    "Student", "Unemployed"
]

options_accident_cause = [
    "Moving Backward", "Overtaking",
    "Changing lane to the left",
    "Changing lane to the right",
    "Overloading",
    "No priority to vehicle",
    "No priority to pedestrian",
    "No distancing",
    "Getting off the vehicle improperly",
    "Improper parking",
    "Overspeed",
    "Driving carelessly",
    "Driving at high speed",
    "Driving to the left",
    "Overturning",
    "Turnover",
    "Driving under the influence of drugs",
    "Drunk driving"
]


# ---------- TITLE ---------- #

st.markdown(
    "<h1 style='text-align:center;'>🚧 Accident Severity Prediction</h1>",
    unsafe_allow_html=True
)


# ---------- IMAGE ---------- #

image = Image.open("doc/theme.png")

col1, col2, col3 = st.columns([1,4,1])
col2.image(image, use_column_width=True)


# ---------- MAIN ---------- #

def main():

    with st.form("prediction_form"):

        st.subheader("Enter Accident Details")

        time = st.selectbox("Time", options_time)
        day = st.selectbox("Day of Week", options_day)
        driver_age = st.selectbox("Driver Age", options_age)

        relation = st.selectbox("Vehicle Driver Relation", options_vehicle_relation)

        experience = st.selectbox("Driving Experience", options_driver_exp)

        vehicle_year = st.selectbox("Vehicle Service Year", options_vehicle_service)

        road_type = st.selectbox("Road Surface Type", options_road_surface)

        road_condition = st.selectbox("Road Surface Condition", options_road_condition)

        light = st.selectbox("Light Condition", options_light)

        collision = st.selectbox("Type of Collision", options_collision)

        vehicles = st.slider("Vehicles Involved", 1, 7)

        casualties = st.slider("Number of Casualties", 1, 8)

        movement = st.selectbox("Vehicle Movement", options_vehicle_move)

        casualty_age = st.selectbox("Casualty Age", options_casualty_age)

        work = st.selectbox("Work of Casualty", options_casualty_work)

        cause = st.selectbox("Cause of Accident", options_accident_cause)

        submit = st.form_submit_button("Predict")


    if submit:

        time = Time_dict[time]
        day = le_Day_of_week.transform([day])
        driver_age = Age_band_of_driver_dict[driver_age]

        relation = le_Vehicle_driver_relation.transform([relation])

        experience = Driving_experience_dict[experience]

        vehicle_year = Service_year_of_vehicle_dict[vehicle_year]

        road_type = le_Road_surface_type.transform([road_type])

        road_condition = le_Road_surface_conditions.transform([road_condition])

        light = Light_conditions_dict[light]

        collision = le_Type_of_collision.transform([collision])

        movement = le_Vehicle_movement.transform([movement])

        casualty_age = Age_band_of_casualty_dict[casualty_age]

        work = le_Work_of_casuality.transform([work])

        cause = le_Cause_of_accident.transform([cause])


        user_input = np.array([
            time,
            *day,
            driver_age,
            *relation,
            experience,
            vehicle_year,
            *road_type,
            *road_condition,
            light,
            *collision,
            vehicles,
            casualties,
            *movement,
            casualty_age,
            *work,
            *cause
        ]).reshape(1,-1)


        prediction = get_prediction(model, user_input)


        if prediction == "Slight injury":
            st.success("Prediction: Slight Injury")

        elif prediction == "Serious Injury":
            st.warning("Prediction: Serious Injury")

        else:
            st.error("Prediction: Fatal Injury")


if __name__ == "__main__":
    main()