import streamlit as st
import joblib

# Load the saved Iris model
iris_LR_model = joblib.load("iris_LR_model.pkl")

# Give a title to the app
st.title('Iris Flower Classification')

# User input section
sepal_length = st.number_input('Enter sepal length:', min_value=0.1, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input('Enter sepal width:', min_value=0.1, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input('Enter petal length:', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
petal_width = st.number_input('Enter petal width:', min_value=0.1, max_value=10.0, value=0.1, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = iris_LR_model.predict(input_data)[0]
    
    if prediction == 0:
        species = "Iris-setosa"
    elif prediction == 1:
        species = "Iris-versicolor"
    else:
        species = "Iris-virginica"
    
    st.success(f"The predicted species: {species}")
