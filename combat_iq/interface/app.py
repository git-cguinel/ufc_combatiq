import streamlit as st
import pandas as pd
import requests

# CSS styles
st.markdown("""
    <style>
        .stButton>button {
            color: white;
            border: 2px solid #4c93af;
            padding: 10px 24px;
            cursor: pointer;
            background-color: #4c93af;
        }
        .stSelectbox>div>div>select {
            background-color: #fff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>UFC Match Winner Prediction</h1>", unsafe_allow_html=True)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.contentapi.ea.com/content/dam/ufc/images/2017/10/easufc3-overview-championshipfighters.jpg.adapt.crop191x100.628p.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# URL of  FastAPI endpoint
api_url = "https://mvp-ef5u7jst5q-ew.a.run.app/predict"


# Load the CSV file
@st.cache
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Load the data
data = load_data('data.csv')

# Concatenate the two columns containing fighter names
all_fighters = pd.concat([data['R_fighter'], data['B_fighter']])

# Remove any duplicate names
fighters_list = sorted(all_fighters.drop_duplicates().tolist())

col1, col2 = st.columns(2, gap = "large")
with col1:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    #st.write("<b><span style='color: white;'>Select Fighter for the red corner</span></b> ", unsafe_allow_html=True)
    red_fighter = st.selectbox("Select Fighter for the red corner",fighters_list)
    st.write("")


with col2:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    #st.write("<b><span style='color: white;'>Select Fighter for the blue corner</span></b> ", unsafe_allow_html=True)
    blue_fighter =st.selectbox("Select Fighter for the blue corner", fighters_list)
    st.write("")


# Your prediction code goes here
if st.button("Predict Winner"):
    # Make a request to the FastAPI endpoint
    response = requests.get(api_url, params={"red_fighter": red_fighter, "blue_fighter": blue_fighter})

    # Check if the request was successful
    if response.status_code == 200:
        prediction_data = response.json()
        fight_outcome = prediction_data['fight_outcome']
        win_rate = prediction_data['win_rate']
        st.markdown(f"<b><span style='color: white;'>The predicted winner is: {fight_outcome}</span></b> ", unsafe_allow_html=True)
        st.markdown(f"<b><span style='color: white;'>Win rate: {win_rate}</span></b>", unsafe_allow_html=True)
    else:
        st.write("Failed to get prediction from the API")
