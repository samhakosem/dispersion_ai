import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def remove_background(image):
    # Placeholder for actual background removal logic
    return image

def detect_prominent_colors(image, num_colors=4):
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image_rgb = image_rgba[:, :, :3].reshape((image_rgba.shape[0] * image_rgba.shape[1], 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_rgb)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def highlight_color_range(image, main_color_bgr, threshold):
    main_color_hsv = cv2.cvtColor(np.uint8([[main_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    lower_bound = np.array([max(0, main_color_hsv[0] - threshold), 50, 50])
    upper_bound = np.array([min(179, main_color_hsv[0] + threshold), 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Create black background
    black_background = np.zeros_like(image)
    highlighted_image = np.where(mask[:, :, np.newaxis] == 0, black_background, result)
    
    return highlighted_image, lower_bound, upper_bound, mask

def calculate_non_transparent_pixels(image):
    """Calculate the number of non-transparent pixels to determine the diamond size."""
    alpha_channel = image[:, :, 3]
    total_diamond_pixels = np.sum(alpha_channel == 255)
    print("Total Non-Transparent Pixels:", total_diamond_pixels)
    return total_diamond_pixels

st.title("Fancy Colored Diamond Grader")
image_no_bg = None
uploaded_file_name = None
uploaded_file = st.file_uploader("Upload an image of the diamond", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_name = uploaded_file.name
    print("file_name", file_name,'uploaded_file_name',uploaded_file_name)
    print('st.session_state',st.session_state)
    print('uploaded_file_name' in st.session_state)
    if ("uploaded_file_name" not in st.session_state.keys()):        
        st.session_state= {}
        st.session_state['uploaded_file_name'] = str(file_name)        
    elif (st.session_state['uploaded_file_name'] != file_name):
        st.session_state= {}
        st.session_state['uploaded_file_name'] = str(file_name)
    if ("image_no_bg" not in st.session_state.keys()):        
        # Read image using PIL to handle transparency
        image = Image.open(uploaded_file)
        image_rgba = np.array(image.convert("RGBA"))

        # Convert the image to OpenCV format
        image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
            
        # Background Removal
        image_no_bg = remove_background(image_bgra)
        st.session_state['image_no_bg'] = image_no_bg        
        # Detect Prominent Colors
        colors = detect_prominent_colors(image_no_bg)
        st.session_state['colors'] = colors
    else:
        colors = st.session_state['colors']
        image_no_bg = st.session_state['image_no_bg']
    st.write("Suggested main colors for the diamond:")

    colors_hex = list(map(lambda color: "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2]), colors))
    colors_hex = list(filter(lambda color_hex: color_hex != "#000000", colors_hex))

    color_hex = colors_hex[0]

    st.session_state['selected_color_hex'] = color_hex
    selected_color_hex = color_hex
    color_hex_list = []
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i, color_hex in enumerate(colors_hex):
        color_hex_list.append(color_hex)
        if cols[i % len(cols)].button(f"PICK", key=color_hex, type='secondary'):
            st.session_state['selected_color_hex'] = color_hex
            selected_color_hex_value = color_hex
            selected_color_hex = color_hex
        cols[i % len(cols)].markdown(f"""
            <div style="
                background-color: {color_hex};
                width: 100px;
                height: 50px;
                border: 2px solid black;
                border-radius: 5px;
                display: inline-block;
                margin: 5px;
            ">{color_hex}</div>
            """, unsafe_allow_html=True)

    selected_color_hex_value = selected_color_hex    
    def hex_to_rgb(hex_value):
        # Remove the '#' if it exists
        hex_value = hex_value.lstrip('#')
        
        # Convert hex to RGB
        rgb = tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
        
        return rgb
    selected_color_bgr = tuple(int(selected_color_hex_value[i:i+2], 16) for i in (1, 3, 5))[::-1]    
    col1, col2 = st.columns(2)
    with col1:
        # Color Picker for Custom Selection
        custom_color_hex = st.color_picker(f"Main Color {selected_color_hex_value}", selected_color_hex_value)
    with col2:
        # Adjust color range threshold
        threshold = st.slider("Adjust color range threshold", 0, 10, 10)

    if selected_color_hex_value:            
        highlighted_image, lower_bound, upper_bound, mask = highlight_color_range(image_no_bg, selected_color_bgr, threshold)

        # Calculate total number of non-transparent pixels
        total_diamond_pixels = calculate_non_transparent_pixels(image_no_bg)

        # Calculate percentage of highlighted pixels
        highlighted_pixels = cv2.countNonZero(mask)
        highlighted_percentage = int(round((highlighted_pixels / total_diamond_pixels) * 100,0))

        st.write(f"<p style='font-size: 24px; font-weight: bold;'>Disperion: {highlighted_percentage}%</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image_no_bg, cv2.COLOR_RGBA2BGRA), caption='Original Image with Background Removed', use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(highlighted_image, cv2.COLOR_RGBA2BGRA), caption='Highlighted Color Range', use_column_width=True)
