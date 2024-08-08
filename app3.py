import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def remove_background(image):
    # Placeholder for actual background removal logic
    return image

def detect_prominent_colors(image, num_colors=4):
    image_rgba = np.array(image).reshape(-1, 4)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_rgba)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def highlight_color_range(image, main_color_rgb, threshold):
    image_rgba = np.array(image)
    lower_bound = np.maximum(main_color_rgb - threshold, 0)
    upper_bound = np.minimum(main_color_rgb + threshold, 255)
    
    mask = np.all(np.logical_and(lower_bound <= image_rgba[:, :, :3], image_rgba[:, :, :3] <= upper_bound), axis=-1)
    highlighted_image = np.zeros_like(image_rgba)
    highlighted_image[mask] = image_rgba[mask]
    
    return Image.fromarray(np.uint8(highlighted_image)), lower_bound, upper_bound, mask

def calculate_non_transparent_pixels(image):
    alpha_channel = np.array(image)[:, :, 3]
    total_diamond_pixels = np.sum(alpha_channel == 255)
    print("Total Non-Transparent Pixels:", total_diamond_pixels)
    return total_diamond_pixels

def hex_to_rgb(hex_value):
    # Remove the '#' if it exists
    hex_value = hex_value.lstrip('#')
    
    # Convert hex to RGB
    rgb = tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
    
    return rgb
st.title("Fancy Colored Diamond Grader")
image_no_bg = None
uploaded_file_name = None
uploaded_file = st.file_uploader("Upload an image of the diamond", type=["jpg", "png", "jpeg"])
def calc_dispersion(highlight_color_range, calculate_non_transparent_pixels, image_no_bg, selected_color_rgb, threshold):
    highlighted_image, lower_bound, upper_bound, mask = highlight_color_range(image_no_bg, np.array(selected_color_rgb), threshold)
        # Calculate total number of non-transparent pixels
    total_diamond_pixels = calculate_non_transparent_pixels(image_no_bg)

        # Calculate percentage of highlighted pixels
    highlighted_pixels = np.sum(mask)
    highlighted_percentage = int(round((highlighted_pixels / total_diamond_pixels) * 100, 0))
    return highlighted_image,highlighted_percentage

if uploaded_file is not None:
    file_name = uploaded_file.name
    if ("uploaded_file_name" not in st.session_state.keys()):        
        st.session_state = {}
        st.session_state['uploaded_file_name'] = str(file_name)        
    elif (st.session_state['uploaded_file_name'] != file_name):
        st.session_state = {}
        st.session_state['uploaded_file_name'] = str(file_name)
    if ("image_no_bg" not in st.session_state.keys()):        
        # Read image using PIL to handle transparency
        image = Image.open(uploaded_file).convert("RGBA")
        image_rgba = np.array(image)

        # Background Removal
        image_no_bg = remove_background(image_rgba)
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

    highlighted_percentages = []
    threshold = st.slider("Adjust color range threshold", 0, 100, 50)
    dispersions=[]
    for i, color_hex in enumerate(colors_hex):
        highlighted_image, highlighted_percentage = calc_dispersion(highlight_color_range, calculate_non_transparent_pixels, image_no_bg, hex_to_rgb(color_hex), threshold)
        dispersions.append(highlighted_percentage)
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
            ">{color_hex}<br> {highlighted_percentage} %</div>
            """, unsafe_allow_html=True)

    selected_color_hex_value = selected_color_hex    

    selected_color_rgb = hex_to_rgb(selected_color_hex_value)        
        # Color Picker for Custom Selection
    custom_color_hex = st.color_picker(f"Main Color {selected_color_hex_value}", selected_color_hex_value)    

    if selected_color_hex_value:            
        highlighted_image, highlighted_percentage = calc_dispersion(highlight_color_range, calculate_non_transparent_pixels, image_no_bg, selected_color_rgb, threshold)
        highlighted_percentages.append(highlighted_percentage)
        
        if highlighted_percentages:
            average_highlighted_percentage = round(np.mean(dispersions),0).astype(int)
            st.write(f"<p style='font-size: 24px; font-weight: bold;'>Dispersion Weighted: {average_highlighted_percentage}% Range: {min(dispersions)}% - {max(dispersions)}% </p>", unsafe_allow_html=True)        

        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.fromarray(image_no_bg), caption='Original Image with Background Removed', use_column_width=True)
        with col2:
            st.image(highlighted_image, caption='Highlighted Color Range', use_column_width=True)

    
