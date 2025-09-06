import streamlit as st
import pickle 
import pandas as pd
import numpy as np
#import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.title("Laptop Price Predictor App")


# brand
company = st.selectbox('Select a Laptop Brand',df['Company'].unique())

# type of laptop
type = st.selectbox("Select a Laptop Type",df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM (GB)',df["Ram"].unique())

# weight
weight = st.number_input('Weight of the Laptop (kg)')

# touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# iPS
ips = st.selectbox('IPS',['NO','Yes'])

# screen size
screen_size = st.number_input('Screen Size (inches)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])          

# hdd
cpu = st.selectbox('CPU',df['cpu_name'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['GPU_brand'].unique())

os = st.selectbox('OS',df['OS'].unique())

# price prediction
if st.button('Predict Laptop Price'):
    #query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips,ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1,12)
    result = np.exp(pipe.predict(query))[0]
    result_formated = "{:,}".format(int(result))
    st.title(f'Predicted Laptop Price Rs. {result_formated}')