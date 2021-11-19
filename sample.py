import streamlit as st
import pandas as pd
import cv2
import numpy as np



menu = ['Home','About me', 'Read Data','Camera']

choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Home':
    st.write('Hello World')
    st.header("First Webapp")

    st.image('media\cool.gif')

    col1,col2 = st.columns(2)

    with col1:
        dog_name = st.text_input('What is your dog name')
        st.write('Your dog name:',dog_name)
    with col2:
        age = st.slider("Dog age",min_value=1,max_value=100)
        st.write('Your dog age:',age)
elif choice == 'Read Data':
     df = pd.read_csv('media\AB_NYC_2019.csv')
     st.dataframe(df)

elif choice == 'About me':
     fileUp = st.file_uploader('Upload file',type =['jpg','png','jpeg'])
     st.image(fileUp)
elif choice == 'Camera':
    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Show Webcam')
    capture_button = st.checkbox('Capture')

    captured_image = np.array(None)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    while run:
        ret, frame = cap.read()        
        # Display Webcam
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB ) #Convert color
        FRAME_WINDOW.image(frame)

        if capture_button:      
            captured_image = frame
            break

    cap.release()




