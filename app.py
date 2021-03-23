import streamlit as st
import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
from PIL import Image , ImageEnhance
from pathlib import Path

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

activities = ["About" ,"License Plate Number Detection"]
choice = st.sidebar.selectbox("Select Activty",activities)
    
if choice =='About':
    st.title("Vehicle License Plate Detection at Security Checkpoints")   
    intro_markdown = read_markdown_file("about.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

if choice == "License Plate Number Detection":
    st.title("Upload Image")
    image_file = st.file_uploader("Upload Image",type=['jpg'])

    if image_file is not None:
            our_image = Image.open(image_file)
            im = our_image.save('out.jpg')
            
            if st.button('Process'):
                img = cv2.imread('out.jpg')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
                bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
                edged = cv2.Canny(bfilter, 30, 200) #Edge detection
                keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(keypoints)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                location = None
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 10, True)
                    if len(approx) == 4:
                        location = approx
                        break
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0,255, -1)
                new_image = cv2.bitwise_and(img, img, mask=mask)
                (x,y) = np.where(mask==255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+1, y1:y2+1]
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)
                #result
                text = result[0][-2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
                st.subheader('PROCESSED GRAY IMAGE')
                st.image(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
               
                st.subheader('PROCESSED EDGE IMAGE')
                st.image(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
                
                st.subheader('PROCESSED IMAGE')
                st.image(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                
                st.subheader('PROCESSED CROPED IMAGE')
                st.image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                
                st.subheader('PROCESSED RESULT IMAGE')
                st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))




            st.subheader('ORIGNAL IMAGE')
            st.image(our_image , use_column_width=True,channels='RGB')

