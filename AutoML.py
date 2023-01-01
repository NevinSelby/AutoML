from operator import index
import streamlit as st
import plotly.express as px
import pycaret
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML ;)")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This website helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    try:
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    except:
        st.error("Please Upload Your Dataset")

if choice == "Modelling":
    try:
        classoreg = st.radio("Choose the type of problem", ["Regression", "Classification"])
        if classoreg == "Regression":
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'): 
                pycaret.regression.setup(df, target=chosen_target, silent=True)
                setup_df = pycaret.regression.pull()
                st.dataframe(setup_df)
                best_model = pycaret.regression.compare_models()
                compare_df = pycaret.regression.pull()
                st.dataframe(compare_df)
                pycaret.regression.save_model(best_model, 'best_model')
        else:
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            if st.button('Run Modelling'): 
                pycaret.classification.setup(df, target=chosen_target, silent=True)
                setup_df = pycaret.classification.pull()
                st.dataframe(setup_df)
                best_model = pycaret.classification.compare_models()
                compare_df = pycaret.classification.pull()
                st.dataframe(compare_df)
                pycaret.classification.save_model(best_model, 'best_model')
    except:
        st.error("Please complete the Uploading and Profiling sections first")

if choice == "Download":
    try:
        with open('best_model.pkl', 'rb') as f: 
            st.download_button('Download Model', f, file_name="best_model.pkl")
    except:
        st.error("Please complete the Modelling section first")
