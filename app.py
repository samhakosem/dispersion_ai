import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title("Interactive Image Color Picker")

# Step 1: Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 2: Display the image using st_canvas
    image = Image.open(uploaded_file)
    width, height = image.size
    image = image.resize((width, height))
    image_np = np.array(image)

    # Initialize or retrieve the list of coordinates
    if 'coordinates' not in st.session_state:
        st.session_state.coordinates = []

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=1,
        background_image=image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode="freedraw",
        key="canvas"
    )

    # Step 3: Get the pixel colors on mouse clicks
    if canvas_result.json_data is not None:
        latest_clicks = canvas_result.json_data["objects"]
        if latest_clicks:
            latest_click = latest_clicks[-1]  # Get the last click only
            if latest_click["type"] == "path":
                # Get the coordinates of the click
                path = latest_click["path"]
                x, y = int(path[0][1]), int(path[0][2])
                if (x, y) not in st.session_state.coordinates:
                    st.session_state.coordinates.append((x, y))
    
    # Display the colors of the selected pixels
    if st.session_state.coordinates:
        # Clear selections button
        if len(st.session_state.coordinates) and st.button("Clear Selections"):
            st.session_state.coordinates = []        
        cols = st.columns(8)
        for i, coord in enumerate(st.session_state.coordinates):
            x, y = coord
            if 0 <= x < width and 0 <= y < height:
                color = image_np[y, x]
                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                with cols[i % 5]:
                    st.markdown(
                        f"<div style='color:white; background-color: rgb({color[0]}, {color[1]}, {color[2]}); height: 20px;'>{color_hex}</div>",
                        unsafe_allow_html=True
                    )
