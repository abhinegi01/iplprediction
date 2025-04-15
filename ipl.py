import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="IPL Score Predictor", layout="centered")
st.title("🏏 IPL Score Prediction")
st.markdown("Predict the total score based on match conditions!")

# Caching training step
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('ipl_data.csv')
    data.columns = data.columns.str.strip()
    data.drop(columns=['date'], inplace=True, errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    data[['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']] = imputer.fit_transform(
        data[['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']]
    )

    columns_to_encode = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
    data = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)

    X = data.drop('total', axis=1)
    y = data['total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, X.columns.tolist(), r2, data

# Load model and data only once
model, model_columns, accuracy, original_data = load_and_train_model()

# Unique values for dropdowns
unique_venues = sorted(original_data.columns[original_data.columns.str.startswith('venue_')])
unique_bat_teams = sorted(original_data.columns[original_data.columns.str.startswith('bat_team_')])
unique_bowl_teams = sorted(original_data.columns[original_data.columns.str.startswith('bowl_team_')])

# Convert column names to user-friendly dropdown options
get_label = lambda col, prefix: col.replace(prefix, '').replace('_', ' ')

venue_options = [get_label(col, 'venue_') for col in unique_venues]
bat_team_options = [get_label(col, 'bat_team_') for col in unique_bat_teams]
bowl_team_options = [get_label(col, 'bowl_team_') for col in unique_bowl_teams]

# Streamlit Inputs
venue = st.selectbox("Select Venue", venue_options)
bat_team = st.selectbox("Select Batting Team", bat_team_options)
bowl_team = st.selectbox("Select Bowling Team", bowl_team_options)

runs = st.number_input("Current Runs", min_value=0, step=1)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)
overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
runs_last_5 = st.number_input("Runs in Last 5 Overs", min_value=0.0, step=0.1)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5, step=1)

# Predict Button
if st.button("Predict Score"):
    try:
        input_data = {
            'runs': runs,
            'wickets': wickets,
            'overs': overs,
            'runs_last_5': runs_last_5,
            'wickets_last_5': wickets_last_5,
        }

        # Add encoded categorical values
        for col in model_columns:
            if col.startswith('venue_'):
                input_data[col] = 1 if venue == get_label(col, 'venue_') else 0
            elif col.startswith('bat_team_'):
                input_data[col] = 1 if bat_team == get_label(col, 'bat_team_') else 0
            elif col.startswith('bowl_team_'):
                input_data[col] = 1 if bowl_team == get_label(col, 'bowl_team_') else 0
            elif col not in input_data:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_columns]  # Ensure correct order

        predicted_total = model.predict(input_df)[0]
        st.success(f"🏆 Predicted Total Score: **{predicted_total:.2f}**")
        st.info(f"📈 Model R² Score: **{accuracy:.2f}**")

    except Exception as e:
        st.error("An error occurred while predicting. Please check the inputs.")
        st.exception(e)
