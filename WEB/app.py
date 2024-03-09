#Code

import streamlit as st 
from PIL import Image
import altair as alt
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import cv2


# Set page title and icon
st.set_page_config(
    page_title="Color Detector",
    page_icon="🌈", 
)


def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_image_pixel(filename):
    with Image.open(filename) as rgb_image:
        image_pixel = rgb_image.getpixel((30, 30))
    return image_pixel

def load_image_with_cv(image_file):
    image = Image.open(image_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += "{:02x}".format(i)
    return hex_color

def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img

def color_analysis(img):
    clf = KMeans(n_clusters=5)
    color_labels = clf.fit_predict(img)
    center_color = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_color = [center_color[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_color[i]) for i in counts.keys()]
    df = pd.DataFrame({'labels': hex_colors, 'counts': counts.values()})
    return df

# ...

def main():
    st.title("Image Colour Analysis") 
    menu = ["Home", "About", "Contact Us"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        
        image_file = st.file_uploader("Upload Image", type=['PNG', 'JPG', 'JPEG'])
        if image_file is not None:
            img = load_image(image_file)
            st.image(img)
            
            # Analysis 
            # Image pixel 
            image_pixel = get_image_pixel(image_file)
            st.write(image_pixel)
            
            # Distribution 
            my_image = load_image_with_cv(image_file)
            modified_image = prep_image(my_image)
            pix_df = color_analysis(modified_image)
            
            p01 = px.pie(pix_df, names='labels', values='counts', color='labels')
            st.plotly_chart(p01)

            # # Display color codes with colors in a box
            # st.subheader("Color Codes with Colors:")
            # for color_code in pix_df['labels']:
            #     st.markdown(f"<div style='background-color: {color_code}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>", unsafe_allow_html=True)
            #     st.write(color_code)

            # Display color codes with colors in a box
            # st.subheader("Color Codes with Colors:")
            # color_boxes = ''.join([f"<div style='background-color: {color_code}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>" for color_code in pix_df['labels']])
            # color_codes = ''.join([f"{color_code}" for color_code in pix_df['labels']])
            # st.markdown(color_boxes, unsafe_allow_html=True)
            # st.write("\n\n\n\n")  # Add some space
            # st.write(color_codes)


            # Display color codes with colors in a box
            st.subheader("Color Codes With Real Colors:")
            color_boxes_and_codes = ''.join([f"<div style='background-color: {color_code}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div> {color_code}<br>" for color_code in pix_df['labels']])
            st.markdown(color_boxes_and_codes, unsafe_allow_html=True)


            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Color Distribution: ")
                st.write(pix_df[['labels', 'counts']])
                
            with col2:
                p02 = px.bar(pix_df, x='labels', y='counts', color='labels')
                st.plotly_chart(p02)


    elif choice == "About": 
        st.title("About Us")

        st.write(
            "1. Welcome to Our Color Detection Application!\n"
            "2. I am Sourin Mukherjee with my team, dedicated to providing you with accurate distribution and colors of your uploaded image\n"
            "3. ⚠️THIS WEBSITE IS USED FOR TRAINING AND DEVELOPMENT PURPOSES⚠️"
            )

        # Insert an image from a local file
        team_image = "Images\DA project members.png"
        st.image(team_image, use_column_width=True)


        st.success("Thank you for choosing our Weather App!")   

    else:
        import requests

        st.title(":mailbox: Get In Touch With Us")
        st.write("Feel free to reach out to us with any questions, feedback, or inquiries.")


        contact_form = """
        <form action="https://formsubmit.co/sourin.mukherjee2105833@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
        </form>
        """
        st.markdown (contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown (f" <style>{f.read()}</style>", unsafe_allow_html=True)
        local_css("style.css")    
    
if __name__ == '__main__':
    main()
