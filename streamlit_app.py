import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = 'media/AB_NYC_2019.csv'

menu = ['Home', 'Read Data', 'Display Image', 'Show Video', 'Capture From Webcam', 'Play Audio', 'About Me']

choice = st.sidebar.selectbox('What puppy can do?', menu)


if choice=='Home':
    st.title("Puppy Wonderland")
    st.header("My First Web App!")

    st.write("")
    st.write("My puppy can do anything!")

    st.image('media/isle_of_dog.gif',
            caption="My lovely black puppy",
            use_column_width='auto')

    col1, col2, col3 = st.columns(3)

    # NAME
    with col1:
        name = st.text_input("Enter your puppy name:")
        if name!="":
            st.write(name, "is a cute name!")

    # AGE
    with col2:
        age = st.slider("Choose your puppy age", min_value=0, max_value=20)
        st.write('Your puppy is ', age, 'years old')

    # FOOD
    with col3:
        food = st.multiselect('What does it eat?', ['Bone', 'Sausage', 'Veggie'])
        if food==['Bone']:
            st.write("He must bark first!")
        elif food==['Sausage']:
            st.write('Quite expensive, but...OK!')
        else:
            st.write('Ohno.... Are you sure?')

elif choice=='Read Data':
    st.title('Hot Dog Summer!')
    st.image('media/dog-beach-lifesaver.png')

    @st.cache()
    def load_data(path):
        return pd.read_csv(path)

    df = load_data(PATH) 
    st.dataframe(df)

    figure, ax = plt.subplots()
    df.groupby('neighbourhood_group')['price'].mean().plot(kind='barh', ax=ax)
    st.pyplot(figure)
    st.write('Amazing chart!')

    price = st.slider('Choose your price', min_value=10, max_value=500)
    filter = df[df['price']<price]
    st.map(filter[['latitude', 'longitude']])

elif choice=='Display Image':
    st.title('My puppy can show images')
    photo_uploaded = st.file_uploader('Upload your best photo here', ['png', 'jpeg', 'jpg'])
    if photo_uploaded!=None:
        image_np = np.asarray(bytearray(photo_uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(image_np, 1)
        st.image(img, channels='BGR')

        st.write(photo_uploaded.size)
        st.write(photo_uploaded.type)

elif choice=='Show Video':
    st.title('Show your puppy best videos here!')
    st.warning('Sounds available on local computer ONLY')
    video_uploaded = st.file_uploader('Upload please', type=['mp4'])
    if video_uploaded!=None:
        st.video(video_uploaded)

if choice == 'Capture From Webcam':
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

elif choice=='Play Audio':
    st.write("Puppy can play music!")
    audio_uploaded = st.file_uploader('Upload your fav song')
    if audio_uploaded!=None:
        audio = audio_uploaded.read()
        st.audio(audio, format='audio/mp3')

elif choice=='About Me':
    st.success('An awesome guy!')
    st.image('media/cool.gif')
    st.balloons()