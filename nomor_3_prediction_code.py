import streamlit as st
import joblib
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

model = joblib.load('2602169926_model_oop.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    numeric_columns,categorical_columns = st.columns(2)

    with numeric_columns:
        st.subheader('Numerical Features')
        Creditscore = st.number_input('CreditScore', min_value=0, max_value=860)
        Age = st.number_input('Age', min_value=15, max_value=95)
        Balance = st.number_input('Balance', min_value=0.0)
        EstimatedSalary = st.number_input('EstimatedSalary', min_value=0.0)
    

    with categorical_columns:
        st.subheader('Categorical Features')
        # Mapping dictionaries for categorical variables
        geography_mapping = {'Spain': 0, 'France': 1, 'Germany': 2}
        gender_mapping = {'Female': 0, 'Male': 1}

        # Add a dropdown menu for categorical prediction
        Geography = ['Spain', 'France', 'Germany']
        selected_category1 = st.selectbox('Select Geography', Geography)
        Gender = ['Female', 'Male']
        selected_category2 = st.selectbox('Select Gender', Gender)

        selected_category1_numeric = geography_mapping[selected_category1]
        selected_category2_numeric = gender_mapping[selected_category2]


        Tenure = [0,1,2,3,4,5,6,7,8,9,10]
        selected_category3 = st.selectbox('Select Tenure', Tenure)

        NumOfProducts = [1,2,3,4]
        selected_category4 = st.selectbox('Select NumOfProducts', NumOfProducts)

        HasCrCard = [0,1]
        selected_category5 = st.selectbox('Select HasCrCard', HasCrCard)

        IsActiveMember = [0,1]
        selected_category6 = st.selectbox('Select IsActiveMember', IsActiveMember)

    
    if st.button('Make Prediction'):
        features = [Creditscore,Age,Balance,EstimatedSalary,selected_category1_numeric,selected_category2_numeric,selected_category3,selected_category4
                    ,selected_category5, selected_category6]
        result = make_prediction(features)

        if result == 0:
            predictions = "Customer will not Churn / not leave the bank" 
        else:
            predictions = "Customer will Churn / leave the bank"

        st.success(f'{predictions}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
