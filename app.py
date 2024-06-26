import streamlit as st
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
import requests
import os

st.title("Diamond Dispersion AI")

class DiamondAreaCalculator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.no_bg_image_path = f'no-bg_{self.image_path}'
        self.img_array = None
        self.total_diamond_pixels = None
        self.load_image()
    
    def remove_backgroup(self):
        print("Removing Background...")
        api_key = os.getenv('REMOVE_BG_API_KEY')
        print("API Key:", api_key)

        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': open(self.image_path, 'rb')},
            data={'size': 'auto'},
            headers={'X-Api-Key': api_key},
        )
        if response.status_code == requests.codes.ok:
            with open(self.no_bg_image_path, 'wb') as out:
                out.write(response.content)
        else:
            print("Error:", response.status_code, response.text)

    
    def load_image(self):
        """Load the image from the file and convert it to a numpy array."""
        try:
            # check if the image has an alpha channel
            has_alpha_channel = None
            tmp_img = Image.open(self.image_path)
            tmp_image_array = np.array(tmp_img)
            if tmp_image_array.shape[2] < 4:
                has_alpha_channel = False
            else:
                has_alpha_channel = True
            if not has_alpha_channel:
                self.remove_backgroup()
                img = Image.open(self.no_bg_image_path)
            else:
                img = tmp_img
            # Enhance the image by increasing saturation
            enhance_image = False  # Set this flag to True if you want to enhance the image, False otherwise
            if enhance_image:
                enhancer = ImageEnhance.Color(img)
                saturation_maximized = enhancer.enhance(5.0)  # Increasing saturation by a factor of 5
                image_array = np.array(saturation_maximized)
            else:
                image_array = np.array(img)
            
            self.img_array = image_array
            adjusted_image = Image.fromarray(image_array)
            print("Image Loaded Successfully.")
            # Save the modified image
            adjusted_image.save(f'intermediate_image_{self.image_path}')
            if self.img_array.shape[2] < 4:
                raise ValueError("Image does not have an alpha channel.")
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def calculate_non_transparent_pixels(self):
        """Calculate the number of non-transparent pixels to determine the diamond size."""
        if self.total_diamond_pixels is None:
            alpha_channel = self.img_array[:, :, 3]
            self.total_diamond_pixels = np.sum(alpha_channel == 255)
        print("Total Non-Transparent Pixels:", self.total_diamond_pixels)
        return self.total_diamond_pixels
    
    def get_all_pixels(self):
        """Get the total number of pixels in the image."""
        return self.img_array.shape[0] * self.img_array.shape[1]
    
    # Optimizing the logical operation for better performance

    def get_marked_area_pixels(self, red_threshold):
        """Calculate the number of marked pixels within the specified color range."""
        # Optimizing the logical operation for better performance
        marked_areas = (self.img_array[:, :, 0] >= red_threshold) & \
                       (self.img_array[:, :, 1] < 50) & \
                       (self.img_array[:, :, 2]  < 50)
        return np.sum(marked_areas)
    
    def calculate_marked_area_percentage(self, red_threshold=150):
        """Calculate the percentage of the diamond area that is marked."""
        if self.total_diamond_pixels is None:
            self.calculate_non_transparent_pixels()
        marked_area_pixels = self.get_marked_area_pixels(red_threshold)
        if self.total_diamond_pixels == 0:
            return 0  # Avoid division by zero
        return round((marked_area_pixels / self.total_diamond_pixels) * 100 , 2)
    
    def generate_marked_area_image(self, red_threshold=150):
        """Generate a new image with only the marked area highlighted."""
        marked_areas = (self.img_array[:, :, 0] >= red_threshold) & \
                       (self.img_array[:, :, 1] < 50) & \
                       (self.img_array[:, :, 2]  < 50)
        marked_area_image = np.zeros(self.img_array.shape, dtype=np.uint8)
        marked_area_image[marked_areas] = self.img_array[marked_areas]
        return Image.fromarray(marked_area_image)

    def process_image(self, red_threshold=150):
        """Perform all the calculations and generate the marked area image."""
        self.calculate_non_transparent_pixels()
        marked_area_percentage = self.calculate_marked_area_percentage(red_threshold)
        marked_area_image = self.generate_marked_area_image(red_threshold)
        print("Percentage of Diamond Area Marked:", marked_area_percentage)
        return marked_area_percentage, marked_area_image


# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image and get resolution
    try:
        image = Image.open(uploaded_file)    
        file_name = uploaded_file.name        
        # dump image to disk
        image.save(file_name)
        diamond_calculator = DiamondAreaCalculator(file_name)    
        marked_area_percentage, marked_area_image = diamond_calculator.process_image()    
        all_pixels = diamond_calculator.get_all_pixels()
        st.write(f"Marked Area Percentage: {marked_area_percentage}%", "Image Resolution:", image.size,"Total Pixels:", all_pixels)
        col1, col2 = st.columns(2)  # Replace st.beta_columns() with st.columns()
        col1.image(marked_area_image, caption="Marked Area Image")
        col2.image(image, caption="Original Image")
    except Exception as e:
        st.warning(f"Error processing image. Make sure to upload image with transparent background.")
else:
    st.info("Please upload an image file.")

