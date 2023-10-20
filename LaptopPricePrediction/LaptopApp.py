import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# Company Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
model = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1448'])

# CPU
cpu = st.selectbox('Brand', df['Cpu Brand'].unique())

# Hard Drive
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
sdd = st.selectbox('SDD(in GB)', [0, 8, 128, 256, 512, 1024])

# gpu
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips =='Yes':
        ips=1
    else:
        ips =0
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
    query = np.array([company,model,ram,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os])

    query =query.reshape(1,12)
    st.title("The Predicted Price of Laptop is"+str(int(np.exp(pipe.predict(query)[0]))))
