#[model]
import pickle
import base64

#[Data Transformation]
import numpy as np
import pandas as pd

#[Dashboard]
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.stylable_container import stylable_container


# Streamlit Page Configuration
st.set_page_config(
    page_title = "Used Car Price Predictor",
    page_icon= "Images/diagram.png",
    layout = "wide",
    initial_sidebar_state= "expanded"
    )

# Title
st.title(":blue[Car Dheko Used Car Price Prediction]")

# Intro
st.write(""" """)
st.image("S:/Car/CarDekho_FeaturedImage_YS.png", width= 800)
#options
Gear = ['4','5','6']

City = ['0','1','2','3','4','5']
City_dict = {'Delhi':2,'Kolkata':5,'Chennai':1,'Hyderabad':3,'Jaipur':4,'Bengaluru':0}



bt = ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
       'Pickup Trucks', 'Convertibles', 'Hybrids', 'Wagon']
bt_dict = {'Hatchback': 0, 'SUV': 7, 'Sedan': 8, 'MUV': 4, 'Coupe': 1,
           'Minivans': 5, 'Pickup': 6, 'Convertibles': 0, 'Hybrids': 3, 'Wagon': 9}
fuel_type = ['Petrol', 'Diesel', 'LPG', 'CNG', 'Electric']
fuel_type_dict = {'Petrol': 4, 'Diesel': 1, 'LPG': 0, 'CNG': 2, 'Electric': 3}

ownerNo = ['0','1','2','3','4','5']

Inusrance_validity = ['Third Party insurance', 'Comprehensive', 'Third Party',
       'Zero Dep', 'Not Available']

Inusrance_validity_dict = {'Third Party insurance': 5, 'Comprehensive': 2, 'Third Party': 4, 
                           'Zero Dep': 6, 'Not Available': 3}


Year_of_Manufacture = ['1','2','3','4','5','6','7','8','9','10',
                       '11','12','13','14','15','16','17','18','19','20','21']

Year_of_Manufacture_dict = {'2018':16,'2017':15,'2016':14,'2019':17,
                            '2021':19,'2020':18,'2015':13,'2014':12,
                            '2022':20,'2013':11,'2012':10,'2011':9,'2010':8,
                            '2009':7,'2023':21,'2008':6,'2007':5,'2006':4,'2004':3,
                            '2005':2,'2003':1,'2002':0}

modelYear = ['1','2','3','4','5','6','7','8','9','10',
             '11','12','13','14','15','16','17','18','19','20','21',
             '22','23','24','25','26','27','28' ]

modelYear_dict = {'2018':23,'2017':22,'2016':21,'2019':24,'2021':26,'2020':25,'2015':20,'2014':19,'2022':27,'2023':28,
                  '2013':18,'2012':17,'2011':16,'2010':15,'2009':14,'2023':28,'2008':13,'2007':12,'2006':11,'2004':9,
                  '2005':10,'2003':8,'2002':7,'2001':6,'1998':3,'1995':1,'1985':0,'1999':4,'2000':5,'1997':2}

transmission = ['Automatic','Manual']
transmission_dict = {'Automatic':0,'Manual':1}

with st.sidebar:
    st.header("Car Details :Cardheko:")
    m_transmission = st.selectbox(label= "Transmission", options= transmission, index= 0, key= "transmission")
    m_Year_of_Manufacture = st.selectbox(label= "Year of Manufacture", options= Year_of_Manufacture_dict, index= 0, key= "Year_of_Manufacture")
    m_modelYear = st.selectbox(label= "Model Year", options= modelYear_dict, index= 0, key= "modelYear")
    m_gear = st.selectbox(label= "Number of gears", options= Gear, index= 0, key= "Gear")
    m_city = st.selectbox(label = 'City Name', options = City_dict, index = 0, key = 'City')
    m_Inusrance_validity = st.selectbox(label= "Insurance Validity", options= Inusrance_validity_dict, index= 0, key= "Inusrance_validity")
    m_ownerNo = st.selectbox(label= "Number of Owners", options= ownerNo, index= 0, key= "ownerNo")
    m_fuel_type = st.selectbox(label= "Fuel Type", options= fuel_type, index= 0, key= "fuel_type")
    m_km = st.number_input(label= "Kilometers Driven", step = 1000, value = 0, key= "km")
    m_bt = st.selectbox(label= "Body Type", options= bt_dict, index= 0, key= "bt")
    m_mileage = st.number_input(label= "Mileage", step = 5, key= "mileage")
    
    
    
    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
            """,
    ):  
        pred_price_button = st.button("Estimate Used Car Price")
        
    
def predict_resale_price():

    # Load pre-trained model
    model = pickle.load(open("S:/Car/random_forest_model.pkl", "rb"))
    

    # Combine user inputs to an array
    user_data = np.array([[int(m_gear),
                           int(m_km),
                           int(m_mileage),
                           int(City_dict.get(m_city)),
                           int(Inusrance_validity_dict.get(m_Inusrance_validity)),
                           int(m_ownerNo),
                           int(fuel_type_dict.get(m_fuel_type)),
                           int(bt_dict.get(m_bt)),
                           int(Year_of_Manufacture_dict.get(m_Year_of_Manufacture)),
                           int(modelYear_dict.get(m_modelYear)),
                           int(transmission_dict.get(m_transmission))
                           ]])

    prediction = model.predict(user_data)

    y_p = prediction.reshape(1, -1)
   # y_predicted_original = scaler.inverse_transform([[1, y_p]])[0][1]

    return f'The estimated used car price is: {prediction[0]:.2f} Lakhs'



if pred_price_button:
    st.write(predict_resale_price())