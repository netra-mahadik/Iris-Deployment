import pandas as pd
import pickle
import streamlit as st
import sklearn

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# streamlit ui
st.write("Iris Flower Classification")
st.write("Enter the features below: ")

# input fields
sl=st.number_input("Sepal Length: ", min_value=4.3, max_value=7.9)
sw=st.number_input("Sepal Width: ", min_value=2, max_value=4.4)
pl=st.number_input("Petal Length: ", min_value=1, max_value=6.9)
pw=st.number_input("Petal Width: ", min_value=0.1, max_value=2.5)

if st.button("Predict"):
    pr=model.predict([[sl,sw,pl,pw]])
    classes=["Setosa","Versicolor","Virginica"]
    st.write(f"Prediction: {classes[pr[0]]}")
