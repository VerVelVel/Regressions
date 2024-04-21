import streamlit as st
import os

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to my Regressions page! 👋")

st.sidebar.success("Select a type of your task regression.")

st.markdown(
    """
    Linear and logistic regression are machine learning methods that make predictions based on the analysis of historical data.

    With this application, you can analyze your data! 🕵️‍♂️

    Use linear regression when your target is a continuous value, and logistic regression when solving a classification problem

    **👈 Select a the type of regression** 
"""
)

