import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

# =============================
# Setup page
# =============================
st.set_page_config(page_title="Morphological Processing Demo", layout="wide")

# Dark/Light mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

mode = "🌞 Light Mode" if not st.session_state.dark_mode else "🌙 Dark Mode"
st.sidebar.button(mode, on_click=toggle_theme)

# CSS custom
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: white; }
        </style>
    """, unsafe_allow_html=True)

st.title("🖼️ Morphological Image Processing Playground")

# =============================
# Sidebar config
# =============================
st.sidebar.header("⚙️ Cài đặt")

operation = st.sidebar.selectbox(
    "Chọn phép biến đổi",
    ["Original", "Binary", "Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
)

kernel_shape = st.sidebar.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"])
kernel_size = st.sidebar.slider("Kernel size", 1, 25, 5, step=2)
iterations = st.sidebar.slider("Iterations", 1, 10, 1)

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

# =============================
# Processing
# =============================
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Kernel shape
    if kernel_shape == "Rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # Morphological operations
    if operation == "Original":
        result = img
    elif operation == "Binary":
        result = binary
    elif operation == "Erosion":
        result = cv2.erode(binary, kernel, iterations=iterations)
    elif operation == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=iterations)
    elif operation == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "Closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "Gradient":
        result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif operation == "Top Hat":
        result = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif operation == "Black Hat":
        result = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

    # =============================
    # UI: Hiển thị ảnh
    # =============================
    col1, col2 = st.columns(2)
    col1.subheader("Ảnh gốc")
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    col2.subheader(f"Kết quả: {operation}")
    if len(result.shape) == 2:
        col2.image(result, channels="GRAY", use_container_width=True)
    else:
        col2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    # =============================
    # Download
    # =============================
    result_pil = Image.fromarray(result if len(result.shape) == 2 else cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Tải ảnh kết quả",
        data=byte_im,
        file_name=f"result_{operation.lower()}.png",
        mime="image/png"
    )
