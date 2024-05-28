import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

# Load the trained model, preprocessor, and top features
model = joblib.load('best_rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
top_features = joblib.load('top_features.pkl')

# Function to load image and convert to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Add custom CSS to style the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content .block-container h2 {
        color: white;
    }
    .sidebar .sidebar-content .block-container h4 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Pages")
page = st.sidebar.selectbox("Go to", ["Home", "Predict"])

# Home page
if page == "Home":
    st.title('Welcome to the Bank Term Deposit Subscription Prediction App')
    st.write('Use this app to predict whether you will subscribe to bank term deposit based on your personal informations.')
    st.write('Go to the Predict page to enter your details and get a prediction.')

    # Add a GIF using base64 encoding
    img_base64 = get_base64_image('dollar.gif')
    st.markdown(f'<img src="data:image/gif;base64,{img_base64}" alt="Bank Term Deposit" style="width:75%;">', unsafe_allow_html=True)
    
# Prediction page
elif page == "Predict":
    st.title('Bank Term Deposit Subscription Prediction')

    # Define input fields for the Streamlit app
    age = st.number_input('Age', min_value=18, max_value=100, value=18)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                           'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.selectbox('Has Credit in Default?', ['no', 'yes'])
    housing = st.selectbox('Has Housing Loan?', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Has Personal Loan?', ['no', 'yes', 'unknown'])
    contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Last Contact Day of the Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.number_input('Last Contact Duration (seconds)', min_value=0, value=0)
    campaign = st.number_input('Number of Contacts During Campaign', min_value=0, value=0)
    pdays = st.number_input('Number of Days Since Last Contact', min_value=0, value=0)
    previous = st.number_input('Number of Contacts Before Campaign', min_value=0, value=0)
    poutcome = st.selectbox('Outcome of the Previous Campaign', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.number_input('Employment Variation Rate', value=0.00)
    cons_price_idx = st.number_input('Consumer Price Index', value=0.00)
    cons_conf_idx = st.number_input('Consumer Confidence Index', value=-0.00)
    euribor3m = st.number_input('Euribor 3 Month Rate', value=0.00)
    nr_employed = st.number_input('Number of Employees', value=0.00)

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'default': [default],
        'housing': [housing], 'loan': [loan], 'contact': [contact], 'month': [month], 'day_of_week': [day_of_week],
        'duration': [duration], 'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate], 'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
    })

    # Add a predict button
    if st.button('Predict'):
        # Preprocess the input data
        input_data_preprocessed = preprocessor.transform(input_data)[:, top_features]
        
        # Make prediction
        prediction = model.predict(input_data_preprocessed)
        
        # Display the prediction
        if prediction[0] == 1:
            st.success('The client is likely to subscribe to a term deposit.')
        else:
            st.warning('The client is not likely to subscribe to a term deposit.')
