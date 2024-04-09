import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

@st.cache_data()
def load_data(data):
    data = pd.read_csv(data)
    return data

data = load_data('Financial_inclusion_dataset.csv')


st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 50px; font-family: Helvetica'>BANKING INCLUSION IN EAST AFRICA (DATA EXPLORATION)</h1>", unsafe_allow_html = True)

st.header('Project Background Information',divider = True)
st.write("The dataset contains demographic information and what financial services are used by approximately 33,600 individuals across East Africa. The ML model role is to predict which individuals are most likely to have or use a bank account. The term financial inclusion means:  individuals and businesses have access to useful and affordable financial products and services that meet their needs – transactions, payments, savings, credit and insurance – delivered in a responsible and sustainable way.")

st.markdown('<br>', unsafe_allow_html = True)
st.write('Below are charts used to explore on the data. The next page(financeapp) allows you to enter relevant parameters about a respondent and the app will predict if the respondent with entered details has an account or otherwise.')

def plotter(dataframe, col1, col2, col3, col4, dep):
    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    st.subheader(f"{col1} vs {dep}")
    col1plot = px.histogram(data_frame= dataframe, x = col1, color = dep)
    st.plotly_chart(col1plot)
    
    st.markdown('<br>',  unsafe_allow_html=True)

    plt.subplot(1,2,2)
    st.subheader(f"{col2} vs {dep}")
    col2plot = px.histogram(data_frame= dataframe, x = col2, color = dep)
    st.plotly_chart(col2plot)
  
    st.markdown('<br>',  unsafe_allow_html=True)

    plt.subplot(2,2,1)
    st.subheader(f"{col3} vs {dep}")
    col3plot = px.histogram(data_frame= dataframe, x = col3, color = dep)
    st.plotly_chart(col3plot)

    st.markdown('<br>',  unsafe_allow_html=True)
    
    plt.subplot(2,2,2)
    st.subheader(f"{col4} vs {dep}")
    col4plot = px.histogram(data_frame = dataframe, x = col4, color = dep)
    st.plotly_chart(col4plot)
    

st.markdown('<br>',  unsafe_allow_html=True)
plotter(data, 'age_of_respondent','household_size', 'marital_status', 'job_type', 'bank_account' )
st.markdown('<br>',  unsafe_allow_html=True)
plotter(data, 'education_level', 'country', 'cellphone_access','location_type', 'bank_account' )


