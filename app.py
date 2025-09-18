import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

from skimage.morphology import skeletonize, convex_hull_image, remove_small_objects, disk
from skimage.filters import rank
from skimage import img_as_ubyte, img_as_float
from scipy.ndimage import binary_fill_holes, laplace, median_filter, grey_opening, grey_closing, grey_dilation, grey_erosion, morphological_gradient

# =============================
# Setup page
# =============================
st.set_page_config(page_title="Morphological Processing Demo", layout="wide")
st.title("🖼️ Morphological Image Processing Playground")

# =============================
# Sidebar config
# =============================
st.sidebar.header("⚙️ Cài đặt")
operation_group = st.sidebar.selectbox(
    "Chọn nhóm phép biến đổi",
    [
        "Original / Binary",
        "Morphology cơ bản",
        "Morphology nâng cao",
        "Khử nhiễu",
        "Tăng cường ảnh"
    ]
)

# -----------------------------
# Morphology cơ bản
# -----------------------------
if operation_group == "Morphology cơ bản":
    morph_mode = st.sidebar.radio(
        "Chọn phép Morphology",
        ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
    )

# -----------------------------
# Morphology nâng cao
# -----------------------------
elif operation_group == "Morphology nâng cao":
    morph_adv_mode = st.sidebar.radio(
        "Chọn phép nâng cao",
        ["Skeletonize", "Convex Hull", "Boundary", "Laplace", "Beucher Gradient"]
    )

# -----------------------------
# Khử nhiễu
# -----------------------------
elif operation_group == "Khử nhiễu":
    denoise_mode = st.sidebar.radio(
        "Chọn phương pháp khử nhiễu",
        ["Median (scipy)", "Median rank (skimage)", "Open+Close", "Fingerprint"]
    )
    if denoise_mode == "Fingerprint":
        finger_mode = st.sidebar.radio(
            "Fingerprint mode",
            ["Opening", "Closing", "Opening + Closing"],
            horizontal=True
        )
    else:
        finger_mode = None

# -----------------------------
# Tăng cường ảnh
# -----------------------------
elif operation_group == "Tăng cường ảnh":
    enhance_mode = st.sidebar.radio(
        "Chọn phép tăng cường",
        ["Local Entropy", "Contrast Enhancement", "Fill Holes", "Remove Small Objects", "Grayscale Erosion", "Grayscale Dilation"]
    )

# =============================
# Extra parameters
# =============================
if operation_group == "Tăng cường ảnh" and enhance_mode == "Local Entropy":
    entropy_radius = st.sidebar.slider("Entropy radius", 1, 20, 5)
if operation_group == "Tăng cường ảnh" and enhance_mode == "Fill Holes":
    fill_structure_size = st.sidebar.slider("Kernel size (structure) để lấp lỗ", 1, 20, 3)
if operation_group == "Tăng cường ảnh" and enhance_mode == "Remove Small Objects":
    min_object_size = st.sidebar.slider("Min object size (pixels)", 1, 1000, 150)

# =============================
# Kernel config
# =============================
kernel_ops = ["Morphology cơ bản", "Morphology nâng cao", "Khử nhiễu", "Tăng cường ảnh"]
use_kernel = operation_group in kernel_ops

if use_kernel:
    kernel_shape = st.sidebar.selectbox("Kernel shape", ["Rectangle", "Ellipse", "Cross"])
    kernel_size = st.sidebar.slider("Kernel size", 1, 25, 5, step=1)
    iterations = st.sidebar.slider("Iterations", 1, 10, 1)
    if kernel_shape == "Rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
else:
    kernel, iterations = None, 1

# =============================
# Upload ảnh
# =============================
uploaded_file = st.file_uploader("📂 Upload một ảnh", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.warning("⚠️ Vui lòng upload ảnh để chạy ứng dụng!")
    st.stop()

file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img is None:
    st.error("⚠️ Không đọc được ảnh upload.")
    st.stop()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary_uint8 = (binary > 0).astype(np.uint8) * 255

# =============================
# Xử lý
# =============================
result = None

# 1. Original / Binary
if operation_group == "Original / Binary":
    mode = st.sidebar.radio("Hiển thị", ["Original", "Binary"])
    result = img if mode == "Original" else binary_uint8

# 2. Morphology cơ bản
elif operation_group == "Morphology cơ bản":
    if morph_mode == "Erosion":
        result = cv2.erode(binary_uint8, kernel, iterations=iterations)
    elif morph_mode == "Dilation":
        result = cv2.dilate(binary_uint8, kernel, iterations=iterations)
    elif morph_mode == "Opening":
        result = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif morph_mode == "Closing":
        result = cv2.morphologyEx(binary_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif morph_mode == "Gradient":
        result = cv2.morphologyEx(binary_uint8, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif morph_mode == "Top Hat":
        result = cv2.morphologyEx(binary_uint8, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif morph_mode == "Black Hat":
        result = cv2.morphologyEx(binary_uint8, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

# 3. Morphology nâng cao
elif operation_group == "Morphology nâng cao":
    if morph_adv_mode == "Skeletonize":
        result = skeletonize(binary_uint8 > 0).astype(np.uint8) * 255
    elif morph_adv_mode == "Convex Hull":
        im_bin = (binary_uint8 > 0)
        chull = convex_hull_image(im_bin)
        chull_diff = img_as_float(chull.copy())
        chull_diff[im_bin] = 2.0
        result = (255 * (chull_diff / chull_diff.max())).astype(np.uint8)
    elif morph_adv_mode == "Boundary":
        erosion = cv2.erode(binary_uint8, kernel, iterations=1)
        result = cv2.subtract(binary_uint8, erosion)
    elif morph_adv_mode == "Laplace":
        result = laplace(binary_uint8)
    elif morph_adv_mode == "Beucher Gradient":
        er = cv2.erode(binary_uint8, kernel, iterations=1)
        di = cv2.dilate(binary_uint8, kernel, iterations=1)
        result = cv2.subtract(di, er)

# 4. Khử nhiễu
elif operation_group == "Khử nhiễu":
    if denoise_mode == "Median (scipy)":
        result = median_filter(gray, size=3)
    elif denoise_mode == "Median rank (skimage)":
        result = rank.median(img_as_ubyte(gray), disk(5))
    elif denoise_mode == "Open+Close":
        oc = cv2.morphologyEx(binary_uint8, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, kernel)
    elif denoise_mode == "Fingerprint":
        im_bin = (binary_uint8 > 0).astype(np.uint8) * 255
        if finger_mode == "Opening":
            result = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)
        elif finger_mode == "Closing":
            result = cv2.morphologyEx(im_bin, cv2.MORPH_CLOSE, kernel)
        else:
            im_o = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(im_o, cv2.MORPH_CLOSE, kernel)

# 5. Tăng cường ảnh
elif operation_group == "Tăng cường ảnh":
    if enhance_mode == "Local Entropy":
        result = rank.entropy(img_as_ubyte(gray), disk(entropy_radius))
    elif enhance_mode == "Contrast Enhancement":
        gray_er = cv2.erode(gray, kernel, iterations=iterations)
        gray_di = cv2.dilate(gray, kernel, iterations=iterations)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result = cv2.add(cv2.subtract(gray_di, gray_er), tophat)
    elif enhance_mode == "Fill Holes":
        im_bin = (binary_uint8 > 0).astype(np.uint8)
        structure = np.ones((fill_structure_size, fill_structure_size))
        result = binary_fill_holes(im_bin, structure=structure).astype(np.uint8) * 255
    elif enhance_mode == "Remove Small Objects":
        im_bin = (binary_uint8 > 0)
        result = remove_small_objects(im_bin, min_size=min_object_size, connectivity=1)
        result = (result.astype(np.uint8)) * 255
    elif enhance_mode == "Grayscale Erosion":
        result = cv2.erode(gray, kernel, iterations=iterations)
    elif enhance_mode == "Grayscale Dilation":
        result = cv2.dilate(gray, kernel, iterations=iterations)

# =============================
# UI: Hiển thị ảnh
# =============================
col1, col2 = st.columns(2)
col1.subheader("Ảnh gốc")
col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

col2.subheader(f"Kết quả: {operation_group}")
if result is not None:
    if result.dtype == bool:
        result = result.astype(np.uint8) * 255
    elif np.issubdtype(result.dtype, np.floating):
        result = (255 * (result - result.min()) / (result.max() - result.min() + 1e-8)).astype(np.uint8)
    elif result.dtype != np.uint8:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(result.shape) == 2:
        col2.image(result, channels="GRAY", use_container_width=True)
    else:
        col2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    result_pil = Image.fromarray(result if len(result.shape) == 2 else cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Tải ảnh kết quả",
        data=byte_im,
        file_name=f"result_{operation_group.lower().replace(' ','_')}.png",
        mime="image/png"
    )