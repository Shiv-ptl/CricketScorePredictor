import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained models
pipe_t20 = pickle.load(open('XGBoost_pipe.pkl','rb'))
pipe_odi = pickle.load(open('XGBoost_odi_pipe.pkl','rb'))

# Define teams and cities options
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Create UI elements
st.title('Cricket Score Predictor')

with st.form(key='prediction_form'):
    batting_team = st.selectbox('Select batting team', sorted(teams))
    bowling_team = st.selectbox('Select bowling team', sorted(teams))
    city = st.selectbox('Select city', sorted(cities))
    match_type = st.radio('Match Type', ['T20s', 'ODIs'], index=0)  # Default to T20s

    col1, col2, col3 = st.columns(3)
    with col1:
        current_score = int(st.number_input('Current Score', step=1))
    with col2:
        overs = st.number_input('Overs done (works for over > 5)', format='%.1f')

    with col3:
        wickets = int(st.number_input('Wickets out',step=1))

    last_five = int(st.number_input('Runs scored in last 5 overs', step =1))

    if st.form_submit_button('Predict Score'):
        # Extract integer and fractional parts of overs
        overs_int = int(overs)
        overs_frac = overs - overs_int

        # Calculate balls left based on overs and match type
        balls_left = 120 - ((overs_int * 6) + int(overs_frac * 10)) if match_type == 'T20s' else 300 - (
                    (overs_int * 6) + int(overs_frac * 10))
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': city,
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets],
            'crr': [crr],
            'last_five': [last_five]
        })

        if match_type == 'T20s':
            result = pipe_t20.predict(input_df)
        else:
            result = pipe_odi.predict(input_df)

        st.header("Predicted Score: " + str(int(result[0])))
