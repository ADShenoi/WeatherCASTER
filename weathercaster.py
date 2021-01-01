import streamlit as st
import pandas as pd
import numpy as np
from pop import customMsg
from Bengaluru import Bengaluru_DTC, Bengaluru_KNC, Bengaluru_RFC, Bengaluru_SVC, Bengaluru_ANN, Bengaluru_line_chart
from Chennai import Chennai_DTC, Chennai_KNC, Chennai_RFC, Chennai_SVC, Chennai_ANN, Chennai_line_chart
from Thiruvananthapuram import Thiruvananthapuram_DTC, Thiruvananthapuram_KNC, Thiruvananthapuram_RFC, Thiruvananthapuram_SVC, Thiruvananthapuram_ANN, Thiruvananthapuram_line_chart
from Jaipur import Jaipur_ANN, Jaipur_DTC, Jaipur_KNC, Jaipur_RFC, Jaipur_SVC, Jaipur_line_chart
from Hyderabad import Hyderabad_ANN, Hyderabad_DTC, Hyderabad_KNC, Hyderabad_RFC, Hyderabad_SVC, Hyderabad_line_chart

st.title('WeatherCaster')
st.sidebar.title('WeatherCaster')
st.markdown('''People have attempted to predict the weather informally for millennia and formally since the 19th century.
 The Climate changing at drastic rate, makes the old weather prediction models hectic and less reliable.\n
 WeatherCaster is a web app providing weather prediction for selected locations.''')
if st.button('App Tour'):
    msg = '''Instructions\n
                       Step 1. Select Location
                       Step 2. Input Temperature, Dew Point, Wind Speed, Humidity and Pressure
                       Step 3. Click on Predict button
                       Step 4. Click on Show Data to view the dataset (Optional)
                       Step 5. Select Prediction Algorithm (Optional)'''
    customMsg(msg, 10, 'warning')

location = st.sidebar.selectbox("Select Location", ['Jaipur', 'Chennai','Bengaluru', 'Thiruvananthapuram', 'Hyderabad'])
model_algo = st.sidebar.selectbox("Prediction Algorithm", ['KNeighborsClassifier', 'ANN', 'DecisionTreeClassifier'
                                                                  , 'RandomForestClassifier', 'SVC'])
model_temp = st.sidebar.slider(label='Tempereature (°F)', min_value=50, max_value=110, value=82)
model_dew = st.sidebar.slider(label='Dew Point (F)', min_value=20, max_value=90, value=77)
model_wind_sp = st.sidebar.slider(label='Wind Speed (mph)', min_value=0, max_value=8, value=6)
model_hum = st.sidebar.slider(label='Humidity (%)', min_value=10, max_value=100, value=92)
model_press = st.sidebar.text_input("Pressure (mph)", 26.70)

Bengaluru_dataset = pd.read_csv('Bengaluru.csv')
Chennai_dataset = pd.read_csv('2020_chennai.csv')
Thiruvananthapuram_dataset = pd.read_csv('finalwithorder.csv')
Jaipur_dataset = pd.read_csv('Jaipur.csv')
Hyderabad_dataset=pd.read_csv('Hyderabad.csv')


Thiruvananthapuram_map = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [8.5241, 76.9366], columns=['lat', 'lon'])
Chennai_map = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [13.0827, 80.2707], columns=['lat', 'lon'])
Bengaluru_map = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [12.9716, 77.5946], columns=['lat', 'lon'])
Jaipur_map = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [26.9124, 75.7873], columns=['lat', 'lon'])
Hyderabad_map = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [17.3850, 78.4867], columns=['lat', 'lon'])


def font_size(x):
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">x</p>', unsafe_allow_html=True)


if location == 'Bengaluru':
    st.map(Bengaluru_map)
elif location == 'Chennai':
    st.map(Chennai_map)
elif location == 'Thiruvananthapuram':
    st.map(Thiruvananthapuram_map)
elif location == 'Jaipur':
    st.map(Jaipur_map)
elif location == 'Hyderabad':
    st.map(Hyderabad_map)

st.markdown('AREA CHART')

if location == 'Chennai':
    Chennai_line_chart(model_temp, model_dew, model_hum, model_wind_sp, model_press)
elif location == 'Hyderabad':
    Hyderabad_line_chart(model_temp, model_dew, model_hum, model_wind_sp, model_press)
elif location == 'Thiruvananthapuram':
    Thiruvananthapuram_line_chart(model_temp, model_dew, model_hum, model_wind_sp, model_press)
elif location == 'Jaipur':
    Jaipur_line_chart(model_temp, model_dew, model_hum, model_wind_sp, model_press)
elif location == 'Bengaluru':
    Bengaluru_line_chart(model_temp, model_dew, model_hum, model_wind_sp, model_press)

if st.sidebar.button('Show Data'):
    if location == 'Bengaluru':
        st.dataframe(Bengaluru_dataset)
    elif location == 'Chennai':
        st.dataframe(Chennai_dataset)
    elif location == 'Thiruvananthapuram':
        st.dataframe(Thiruvananthapuram_dataset)
    elif location == 'Jaipur':
        st.dataframe(Jaipur_dataset)
    elif location == 'Hyderabad':
        st.dataframe(Hyderabad_dataset)

if st.sidebar.button('Predict'):
    if location == 'Bengaluru':
        if model_algo == 'KNeighborsClassifier':
            predicted = Bengaluru_KNC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'ANN':
            predicted = Bengaluru_ANN([float(model_temp), float(model_dew), float(model_hum), float(model_wind_sp), float(model_press)])
        elif model_algo == 'DecisionTreeClassifier':
            predicted = Bengaluru_DTC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'RandomForestClassifier':
            predicted = Bengaluru_RFC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'SVC':
            predicted = Bengaluru_SVC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
    elif location == 'Chennai':
        if model_algo == 'KNeighborsClassifier':
            predicted = Chennai_KNC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'ANN':
            predicted = Chennai_ANN([float(model_temp), float(model_dew), float(model_hum), float(model_wind_sp), float(model_press)])
        elif model_algo == 'DecisionTreeClassifier':
            predicted = Chennai_DTC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'RandomForestClassifier':
            predicted = Chennai_RFC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'SVC':
            predicted = Chennai_SVC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
    elif location == 'Thiruvananthapuram':
        if model_algo == 'KNeighborsClassifier':
            predicted = Thiruvananthapuram_KNC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'ANN':
            predicted = Thiruvananthapuram_ANN([float(model_temp), float(model_dew), float(model_hum), float(model_wind_sp), float(model_press)])
        elif model_algo == 'DecisionTreeClassifier':
            predicted = Thiruvananthapuram_DTC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'RandomForestClassifier':
            predicted = Thiruvananthapuram_RFC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'SVC':
            predicted = Thiruvananthapuram_SVC([model_temp, model_dew, model_hum, model_wind_sp, model_press])

    elif location == 'Jaipur':
        if model_algo == 'KNeighborsClassifier':
            predicted = Jaipur_KNC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'ANN':
            predicted =  Jaipur_ANN([float(model_temp), float(model_dew), float(model_hum), float(model_wind_sp), float(model_press)])
        elif model_algo == 'DecisionTreeClassifier':
            predicted = Jaipur_DTC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'RandomForestClassifier':
            predicted = Jaipur_RFC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'SVC':
            predicted = Jaipur_SVC([model_temp, model_dew, model_hum, model_wind_sp, model_press])

    elif location == 'Hyderabad':
        if model_algo == 'KNeighborsClassifier':
            predicted = Hyderabad_KNC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'ANN':
            predicted = Hyderabad_ANN([float(model_temp), float(model_dew), float(model_hum), float(model_wind_sp), float(model_press)])
        elif model_algo == 'DecisionTreeClassifier':
            predicted = Hyderabad_DTC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'RandomForestClassifier':
            predicted = Hyderabad_RFC([model_temp, model_dew, model_hum, model_wind_sp, model_press])
        elif model_algo == 'SVC':
            predicted = Hyderabad_SVC([model_temp, model_dew, model_hum, model_wind_sp, model_press])

    st.write('The atmosphere of ', location, ' will mostly be ', *predicted, '.')
print(pd.__version__)
