import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = 'Finance App',
    page_icon = '***'
)
st.title('Main Page')
st.sidebar.success('Select a page above')

@st.cache_data()
def load_data(data):
    data = pd.read_csv(data)
    return data

data = load_data('Financial_inclusion_dataset.csv')

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 60px; font-family: Helvetica'>BANKING INCLUSION PREDICTION APP IN EAST AFRICA</h1>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Kunle Odukoya</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.image('pngwing.com (5).png', width = 400,  caption = 'Financial Inclusion in East Africa Project')


respondent_age = st.sidebar.slider('Please Enter Age of Respondent', min_value = data['age_of_respondent'].min()-1, max_value = data['age_of_respondent'].max()+1)
householdsize = st.sidebar.slider('Please Enter Household size', min_value = data['household_size'].min()-1, max_value = data['household_size'].max()+1)

job = data['job_type'].unique()
jobtype = st.sidebar.selectbox('Enter Job Type', options = job)

Education = data['education_level'].unique()
educationlevel = st.sidebar.selectbox('Enter your level of education', options = Education)

marital_status = data['marital_status'].unique()
maritalstatus = st.sidebar.selectbox('Enter your marital stats', options = marital_status)

country = data['country'].unique()
Country = st.sidebar.selectbox('Enter your country', options = country)

st.markdown("<br>", unsafe_allow_html=True)

#Creating a dataframe with user inputs
user_input = pd.DataFrame()
user_input['age_of_respondent'] = [respondent_age]
user_input['household_size'] = [householdsize]
user_input['job_type'] = [jobtype]
user_input['education_level'] = [educationlevel]
user_input['marital_status'] = [maritalstatus]
user_input['country'] = [Country]

st.markdown("<br>", unsafe_allow_html=True)
st.header('Input Variables', divider = True)
st.dataframe(user_input, use_container_width = True)

#Downloading models
model_job = pickle.load(open('job_type_pickleencoder.pkl', 'rb'))
model_education = pickle.load(open('education_level_pickleencoder.pkl', 'rb'))
model_marital = pickle.load(open('marital_status_pickleencoder.pkl', 'rb'))
model_country = pickle.load(open('country_pickleencoder.pkl', 'rb'))
model_rfn = pickle.load(open('model_picklerfn.pkl', 'rb'))

#Transforming responses with above models
user_input['job_type'] = model_job.transform([[jobtype]])
user_input['education_level'] = model_education.transform([[educationlevel]])
user_input['marital_status'] = model_marital.transform([[maritalstatus]])
user_input['country'] = model_country.transform([[Country]])

st.markdown("<br>", unsafe_allow_html=True)
st.header('Transformed Variables', divider = True)
st.dataframe(user_input, use_container_width = True)

x = user_input.copy()

st.markdown("<br>", unsafe_allow_html=True)

c = st.container(height = 200, border = True)
c.markdown("<h1 style = 'color: #FF9800; text-align: center; font-size: 40px; font-family: Helvetica'>PREDICTION OUTCOME</h1>", unsafe_allow_html = True)

def predict():
    outcome = model_rfn.predict(x)

    if outcome == 0:
        c.markdown("<h1 style = 'color: #A0153E; text-align: center; font-size: 20px; font-family: Helvetica'>CLIENT DOES NOT HAVE AN ACCOUNT</h1>", unsafe_allow_html = True)
    else:
        c.markdown("<h1 style = 'color: #5356FF; text-align: center; font-size: 20px; font-family: Helvetica'>CLIENT HAS AN ACCOUNT</h1>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)
st.button('Predict', on_click = predict)


