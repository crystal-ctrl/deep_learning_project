import streamlit as st
from PIL import Image
import numpy as np
import cv2
import keras
from keras.models import load_model
from scipy.spatial import distance
# from streamlit_webrtc import webrtc_streamer

################
##   Tiltle   ##
################
# app = MultiApp()
image = Image.open('MaskPatrol.png')
col1, col2, col3= st.beta_columns([2,4,8])
with col1:
    st.write("")
with col2:
    st.write("")
    st.write("")
    st.image(image)
with col3:
    st.title("Mask Patrol")
    st.markdown("If you must **mask**, I shall answer...")
st.markdown("---")

st.sidebar.markdown("Mask detection on:")
choice = st.sidebar.selectbox("", ["Home","Image","Webcam"])

################
##    model   ##
################
# Load the model
model = load_model("binary_model.h5")
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict(img):
    # img = cv2.imread("./images/out.jpg")
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img,scaleFactor=1.1, minNeighbors=8)

    if len(faces) > 0:
        out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
        # resize image
        desired_height=1000
        img_height = img.shape[0]
        scale = desired_height / img_height
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        out_img = cv2.resize(out_img, dim, interpolation = cv2.INTER_AREA)

        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

            crop = out_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(150,150))
            crop = np.reshape(crop,[1,150,150,3])/255.0
            mask_result = model.predict_classes(crop)

            if mask_result == 0:
                cv2.putText(out_img,"With Mask",(x, y-10), cv2.FONT_HERSHEY_DUPLEX,1,(102,204,0),2)
                cv2.rectangle(out_img,(x,y),(x+w,y+h),(102,204,0),5)
            elif mask_result == 1:
                cv2.putText(out_img,"No Mask",(x, y-10), cv2.FONT_HERSHEY_DUPLEX,1,(255,51,51),2)
                cv2.rectangle(out_img,(x,y),(x+w,y+h),(255,51,51),5)

        # out_img = cv.cvtColor(out_img, cv.COLOR_BGR2RGB)
        return out_img
    else:
        print("No Face!")

################
##    Home    ##
################
if choice == "Home":
    col1, col2, col3= st.beta_columns([1,8,1])
    with col1:
        st.write("")
    with col2:
        st.title('A Face Mask Detection System')
        st.subheader('Built with OpenCV and Keras/TensorFlow leveraging Deep Learning and Computer Vision Concepts to detect face mask in still images as well as in real-time webcam streaming.')
        st.write('You can choose the options from the left.')
        st.write("")
    with col3:
        st.write("")
    col1, col2, col3= st.beta_columns([3,6,2])
    with col1:
        st.write("")
    with col2:
        st.header('Upcoming Features: ')
        st.markdown("- Webcam Mask Detection")
        st.markdown("- Detecting Incorrect Mask")
    with col3:
        st.write("")
################
##    Image   ##
################
if choice == "Image":
    st.subheader('Upload the image for detection')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"]) #upload image
    if uploaded_file is not None:
        image = Image.open(uploaded_file) #making compatible to PIL
        # image = np.array(Image.open(uploaded_file))
        image = image.save('./images/out.jpg')
        img = cv2.imread("./images/out.jpg")
        st.write("")
        st.write("**Image uploaded successfullly!**", use_column_width=True)
        if st.button("Detect"):
            out_img = predict(img)
            st.image(out_img, caption="Processed Image", use_column_width=True)
    else:
        cover = Image.open('cover image.jpeg')
        st.image(cover, caption="Mask me an Image", use_column_width=True)

################
##   Webcam   ##
################
if choice == "Webcam":
    st.subheader('Real-time mask checking...')
    # webrtc_streamer(key="example")
    # st.markdown("This feature will be available soon...")
    run = st.checkbox('Open Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        # Reading image from video stream
        _, img = camera.read()
        # Call method we defined above
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = predict(img)
          # st.image(img, use_column_width=True)
        FRAME_WINDOW.image(img)
    if not run:
        st.write('Webcam has stopped.')
