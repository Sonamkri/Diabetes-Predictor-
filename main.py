import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def Normalize(data):
    """
    Normalize the input data using pre-defined mean and scale.
    """
    new_scaler = StandardScaler()

    mean = [3.8990228, 121.93648208, 69.06026059, 20.25895765, 82.04885993, 31.81042345, 0.4754544, 33.21335505]
    scale = [3.37835113, 31.80325621, 19.49988601, 16.17845441, 120.07301344, 7.81549589, 0.34034154, 11.5988213]

    new_scaler.mean_ = mean
    new_scaler.scale_ = scale

    # Transform the data
    data = np.array(data).reshape(1, -1)
    x_new_scaled = new_scaler.transform(data)

    return np.array(x_new_scaled).reshape(1, -1)


def Diabetes_Predictor(data):
    """
    Predict whether a person is diabetic using the trained model.
    """
    loaded_model = pickle.load(open('diabetes_model.pkl', 'rb'))
    test = Normalize(data)
    res = loaded_model.predict(test).round()
    return "🩺 **The person is Diabetic!**" if res == 1 else "✅ **The person is Not Diabetic!**"


def main():
    """
    Main function for the Streamlit app.
    """
    # Page title with color
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🌟 Diabetes Predictor 🌟</h1>", unsafe_allow_html=True)
    st.image("Gemini_Generated_Image_5lw66j5lw66j5lw6.jpeg", caption="Predict Diabetes with Confidence", use_column_width=True)

    st.markdown("<h3 style='color: #007BFF;'>📋 Enter your medical details below:</h3>", unsafe_allow_html=True)

    # Input fields for user data with placeholder hints
    Pregnancies = st.text_input("🤰 Number of Pregnancies", placeholder="e.g., 2")
    Glucose = st.text_input("🍭 Glucose Level", placeholder="e.g., 120")
    BloodPressure = st.text_input("💓 Blood Pressure", placeholder="e.g., 80")
    SkinThickness = st.text_input("📏 Skin Thickness", placeholder="e.g., 20")
    Insulin = st.text_input("💉 Insulin Level", placeholder="e.g., 85")
    BMI = st.text_input("⚖️ BMI", placeholder="e.g., 25.0")
    DiabetesPedigreeFunction = st.text_input("🧬 Diabetes Pedigree Function", placeholder="e.g., 0.5")
    Age = st.text_input("🎂 Age", placeholder="e.g., 30")

    st.markdown("<hr style='border: 1px solid #FF5733;'>", unsafe_allow_html=True)

    # Prediction button
    diagnosis = ''
    if st.button("🔍 Get Test Result"):
        try:
            # Convert inputs to floats
            inputs = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diagnosis = Diabetes_Predictor(inputs)
        except ValueError:
            diagnosis = "⚠️ **Please provide valid numerical inputs for all fields!**"

    st.markdown(f"<h3 style='color: #28A745;'>{diagnosis}</h3>", unsafe_allow_html=True)

    # Sidebar information
    st.sidebar.header("About the Model")
    st.sidebar.markdown("""
    <div style="background-color: #FDEDEC; padding: 10px; border-radius: 10px;">
        <p style="font-size: 16px; color: #C0392B;"><strong>This model predicts whether a person is diabetic based on medical input data.</strong></p>
        <ul>
            <li>🔬 Uses an <strong>Artificial Neural Network (ANN)</strong>.</li>
            <li>📈 Accuracy: <strong>76%</strong>.</li>
            <li>📝 Developed for quick, easy, and reliable predictions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    ### Instructions:
    - Fill in all required fields in the main section.
    - Click **Get Test Result** to view the prediction.
    """)
    st.sidebar.markdown("**Disclaimer:** This is a demo app. Always consult a medical professional for accurate diagnosis.")


if __name__ == '__main__':
    main()
