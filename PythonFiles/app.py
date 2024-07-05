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
Comfort_and_convenience = ['1','2','3','4','5','6','7','8','9','10',
                           '11','12','13','14','15','16','17','18','19','20',
                           '21','22','23','24','25','26','27','28','29','30',
                           '31','32','33','34','35','36','37']


Safety = ['1','2','3','4','5','6','7','8','9','10',
          '11','12','13','14','15','16','17','18','19','20',
          '21','22','23','24','25','26','27','28','29','30',
          '31','32','33','34','35','36','37','38','39','40',
          '41','42','43','44','45','46','47','48']

Exterior = ['1','2','3','4','5','6','7','8','9','10',
            '11','12','13','14','15','16','17','18','19','20',
            '21','22','23','24','25','26','27']


Interior = ['1','2','3','4','5','6','7','8','9','10',
            '11','12','13','14','15','16','17','18']

Gear = ['1','2','3','4','5','6','7','8','9']

Entertainment_and_communication = ['1','2','3','4','5','6','7','8','9','10',
                                   '11','12','13','14','15','16','17','18','19','20','21']

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
    m_Comfort_and_convenience = st.selectbox(label= "Comfort and Convenience", options= Comfort_and_convenience, index= 0, key= "Comfort_and_convenience")
    m_Safety = st.selectbox(label= "Safety", options= Safety, index= 0, key= "Safety")
    m_Exterior = st.selectbox(label= "Exterior", options= Exterior, index= 0, key= "Exterior")
    m_Interior = st.selectbox(label= "Interior", options= Interior, index= 0, key= "Interior")
    m_Gear = st.selectbox(label= "Gear", options= Gear, index= 0, key= "Gear")
    m_Entertainment_and_communication = st.selectbox(label= "Entertainment and Communication", options= Entertainment_and_communication, index= 0, key= "Entertainment_and_communication")
        
    
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
    user_data = np.array([[int(m_Comfort_and_convenience),
                           int(m_Safety),
                           int(m_Exterior),
                           int(m_Interior),
                           int(m_Gear),
                           int(m_Entertainment_and_communication),
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