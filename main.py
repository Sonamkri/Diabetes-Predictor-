import streamlit as st 
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



def Normalize(data):
    #Re-create the scaler with the same parameters
    new_scaler = StandardScaler()

    mean = [  3.8990228 , 121.93648208 , 69.06026059 , 20.25895765 , 82.04885993,31.81042345,   0.4754544 ,  33.21335505]
    scale = [  3.37835113 , 31.80325621 , 19.49988601,  16.17845441 ,120.07301344,7.81549589 ,  0.34034154 , 11.5988213 ]

    new_scaler.mean_ = mean
    new_scaler.scale_ = scale

    #Transforming
    data = np.array(data).reshape(1,-1)
    x_new_scaled = new_scaler.transform(data)

    # Now you can use new_scaler to transform other data
    return (np.array(x_new_scaled).reshape(1,-1))

# Daibetes Predictor
def Daibetes_Predictor(data):
    loaded_model = pickle.load(open('diabetes_model.pkl','rb'))
    test = Normalize(data)
    res = loaded_model.predict(test).round()
    if(res == 1):
        return "Person is Diabetic"
    else:
        return "person is Not Diabetic"

# Main Function
def main():
    st.title("Daibetes Predictor")

    # Taking inputs
    Pregnancies = st.text_input('Enter the Number of Pregnancies')
    Glucose = st.text_input('Enter Glucose Level')
    BolldPressure = st.text_input('Enter Blood Pressure Value')
    SkinThickness = st.text_input('Enter Skin Thickness')
    Insulin = st.text_input('Enter Insulin Level')
    BMI = st.text_input('Enter BMI')
    DiabetesPedigreeFunction = st.text_input('Enter Diabetes Pedigree Function Value')
    Age = st.text_input('Enter your text age')

    
    diagnosis = ''
    # Creating Button For Prediction 
    if st.button('Diabetes Test Result'):
        diagnosis = Daibetes_Predictor([Pregnancies,Glucose,BolldPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age ])

    st.success(diagnosis)

    

    # Description of the model
    st.sidebar.write(""" 
    ## **Description of the model....** 
    """)
    st.sidebar.write("""
    ### This model predicts whether a person is diabetic based on medical input data.
    It uses an Artificial Neural Network (ANN) with an accuracy of **76%**.
    Please enter the relevant details below to check the prediction.
    """)


# Main Function Execution....
if __name__ == '__main__':
    main()
