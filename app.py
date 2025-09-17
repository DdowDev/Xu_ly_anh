import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import skimage.morphology as morph
import skimage.filters as filters
import skimage.exposure as exposure
import skimage.measure as measure
import skimage.util as util
import skimage.feature as feature
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters.rank import median
from skimage.morphology import disk, square, rectangle, diamond
import matplotlib.pyplot as plt

st.set_page_config(page_title="Morphology Studio", layout="wide")
st.title("🧩 Morphology Studio — Morphological Image Processing (Advanced)")

# Initialize session state for storing results
if 'result' not in st.session_state:
    st.session_state.result = None
if 'intermediates' not in st.session_state:
    st.session_state.intermediates = []
if 'img_cv' not in st.session_state:
    st.session_state.img_cv = None

# --------------------------
# Sidebar: settings & upload
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Theme", ("Light", "Dark"), index=0)
    if mode == "Dark":
        st.markdown(
            """<style>
            .reportview-container { background: #0E1117; color: #E6EDF3; }
            </style>""",
            unsafe_allow_html=True,
        )
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    st.markdown("### Kernel & Iteration")
    kernel_size = st.slider("Kernel size (odd)", 1, 51, 5, step=2)
    kernel_shape = st.selectbox("Kernel shape", ["Rectangle", "Disk", "Diamond", "Square"])
    iterations = st.slider("Iterations", 1, 10, 1)
    st.markdown("---")
    st.markdown("### Extras")
    show_pipeline = st.checkbox("Show intermediate pipeline steps", value=True)
    blend_alpha = st.slider("Comparison blend (original↔result)", 0.0, 1.0, 0.5, step=0.1)  # Increased step for better UX
    st.markdown("---")
    st.markdown("## Quick demo images")
    st.write("Bạn có thể dùng ảnh demo nếu chưa có ảnh riêng")
    if st.button("Use demo fingerprint"):
        uploaded = "demo_fingerprint"
    if st.button("Use demo text"):
        uploaded = "demo_text"

# --------------------------
# Helper functions
# --------------------------
def pil_to_cv(img_pil):
    arr = np.array(img_pil)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def read_image(uploaded_obj):
    if isinstance(uploaded_obj, str) and uploaded_obj.startswith("demo_"):
        name = uploaded_obj
        if name == "demo_fingerprint":
            demo = Image.open(BytesIO(_demo_fingerprint_bytes())).convert("RGB")
            return pil_to_cv(demo)
        if name == "demo_text":
            demo = Image.open(BytesIO(_demo_text_bytes())).convert("RGB")
            return pil_to_cv(demo)
    if uploaded_obj is None:
        return None
    try:
        img = Image.open(uploaded_obj).convert("RGB")
        return pil_to_cv(img)
    except Exception as e:
        st.error(f"Không thể đọc ảnh: {str(e)}")
        return None

# demo image bytes (simple generated)
def _demo_text_bytes():
    canvas = np.zeros((300,600,3), dtype=np.uint8) + 255
    cv2.putText(canvas, "OpenCV Morphology", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()

def _demo_fingerprint_bytes():
    h,w = 400,300
    img = np.zeros((h,w), dtype=np.uint8)
    for i in range(h):
        offset = int(10*np.sin(i/10.0))
        cv2.line(img, (0,i),(w-1,i+offset), color=255, thickness=1)
    img = cv2.GaussianBlur(img, (5,5), 1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def make_structuring_element(shape, k):
    if shape == "Rectangle":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    if shape == "Square":
        return square(k)
    if shape == "Disk":
        return disk(k//2)
    if shape == "Diamond":
        return diamond(k//2)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))

def ensure_gray_uint8(img_cv):
    if img_cv.ndim == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()
    return img_as_ubyte(gray / 255.0) if gray.max() > 255 else img_as_ubyte(gray/255.0) if gray.max()<=1 else gray

# --------------------------
# Core morphological operations
# --------------------------
def op_erosion(img_bin, k, iters=1):
    kernel = np.ones((k,k), np.uint8)
    return cv2.erode(img_bin, kernel, iterations=iters)

def op_dilation(img_bin, k, iters=1):
    kernel = np.ones((k,k), np.uint8)
    return cv2.dilate(img_bin, kernel, iterations=iters)

def op_opening(img_bin, k, iters=1):
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=iters)

def op_closing(img_bin, k, iters=1):
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=iters)

def op_gradient(img_bin, k):
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel)

def op_tophat(img_gray, k):
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)

def op_blackhat(img_gray, k):
    kernel = np.ones((k,k), np.uint8)
    return cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

def remove_small_objects_bin(img_bin, min_size=64):
    lbl = util.img_as_bool(img_bin)
    cleaned = morph.remove_small_objects(lbl, min_size=min_size)
    return (cleaned.astype(np.uint8) * 255)

def convex_hull_bin(img_bin):
    lbl = util.img_as_bool(img_bin)
    hull = morph.convex_hull_image(lbl)
    return (hull.astype(np.uint8) * 255)

def fill_holes_bin(img_bin):
    lbl = util.img_as_bool(img_bin)
    filled = morph.remove_small_holes(lbl, area_threshold=5000)
    return (filled.astype(np.uint8) * 255)

def extract_boundary_bin(img_bin):
    k = 3
    d = op_dilation(img_bin, k, iters=1)
    e = op_erosion(img_bin, k, iters=1)
    return cv2.subtract(d, e)

def skeletonize_bin(img_bin):
    lbl = util.img_as_bool(img_bin)
    skel = morph.skeletonize(lbl)
    return (skel.astype(np.uint8) * 255)

def median_filter_gray(img_gray, radius=3):
    se = disk(radius)
    img_u8 = img_as_ubyte(img_gray/255.0) if img_gray.max() <= 1 else img_gray.astype(np.uint8)
    med = median(img_u8, se)
    return med

def morphological_laplace(img_gray, k):
    grad = op_gradient(img_gray, k) if img_gray.ndim==2 else op_gradient(cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY), k)
    lap = cv2.Laplacian(grad, cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap)
    return lap_u8

def beucher_gradient(img_bin, k):
    d = op_dilation(img_bin, k)
    e = op_erosion(img_bin, k)
    return cv2.subtract(d, e)

def local_entropy_map(img_gray, radius=7):
    from skimage.filters.rank import entropy
    se = disk(radius)
    img_u8 = img_as_ubyte(img_gray/255.0) if img_gray.max() <=1 else img_gray.astype(np.uint8)
    ent = entropy(img_u8, se)
    ent_u8 = img_as_ubyte(ent/ent.max()) if ent.max()>0 else (ent.astype(np.uint8))
    return ent_u8

# --------------------------
# Build UI pipeline
# --------------------------
if uploaded is None:
    st.info("Upload một ảnh hoặc chọn demo ở sidebar để bắt đầu.")
    st.stop()

img_cv = read_image(uploaded)
if img_cv is None:
    st.error("Không đọc được ảnh. Kiểm tra file.")
    st.stop()

# Store the image in session state
st.session_state.img_cv = img_cv

# Show original
col0, col1 = st.columns([1,2])
with col0:
    st.subheader("Ảnh gốc")
    st.image(cv_to_pil(img_cv), use_container_width=True)

# Options for operations (multi-select pipeline)
st.subheader("🔁 Chọn pipeline (thứ tự sẽ áp dụng từ trên xuống)")
ops = st.multiselect(
    "Chọn các bước (có thể chọn nhiều) — kéo để sắp xếp thứ tự",
    [
        "Convert to Grayscale",
        "Median Denoise",
        "Binary Threshold (Otsu)",
        "Erosion",
        "Dilation",
        "Opening",
        "Closing",
        "Remove Small Objects",
        "Fill Holes",
        "Extract Boundary",
        "Skeletonize",
        "Convex Hull",
        "Top Hat (contrast)",
        "Black Hat (contrast)",
        "Beucher Gradient",
        "Morphological Laplace",
        "Local Entropy Map",
    ],
    default=["Binary Threshold (Otsu)"]
)

# Params for some ops
st.markdown("### Params for specific ops")
min_obj = st.number_input("Min object size (px) for Remove Small Objects", min_value=1, value=150)
entropy_radius = st.slider("Local entropy radius", 3, 31, 7)
median_radius = st.slider("Median filter radius", 1, 5, 2)
fill_holes_area = st.number_input("Fill holes area_threshold", min_value=1, value=5000)

# Run pipeline button
if st.button("Run Pipeline"):
    # Reset previous results
    st.session_state.result = None
    st.session_state.intermediates = []

    # Prepare grayscale & binary
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.ndim==3 else img_cv.copy()
    current = img_cv.copy()

    # Run pipeline
    for op in ops:
        if op == "Convert to Grayscale":
            current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim==3 else current
            st.session_state.intermediates.append(("Grayscale", current.copy()))
        elif op == "Median Denoise":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            med = median_filter_gray(g, median_radius)
            current = med
            st.session_state.intermediates.append(("Median", current.copy()))
        elif op == "Binary Threshold (Otsu)":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            current = th
            st.session_state.intermediates.append(("Binary Otsu", current.copy()))
        elif op == "Erosion":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_erosion(cur_bin, kernel_size, iterations)
            st.session_state.intermediates.append(("Erosion", current.copy()))
        elif op == "Dilation":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_dilation(cur_bin, kernel_size, iterations)
            st.session_state.intermediates.append(("Dilation", current.copy()))
        elif op == "Opening":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_opening(cur_bin, kernel_size, iterations)
            st.session_state.intermediates.append(("Opening", current.copy()))
        elif op == "Closing":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_closing(cur_bin, kernel_size, iterations)
            st.session_state.intermediates.append(("Closing", current.copy()))
        elif op == "Remove Small Objects":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = remove_small_objects_bin(cur_bin, min_obj)
            st.session_state.intermediates.append(("RemoveSmall", current.copy()))
        elif op == "Fill Holes":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = fill_holes_bin(cur_bin)
            st.session_state.intermediates.append(("FillHoles", current.copy()))
        elif op == "Extract Boundary":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = extract_boundary_bin(cur_bin)
            st.session_state.intermediates.append(("Boundary", current.copy()))
        elif op == "Skeletonize":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = skeletonize_bin(cur_bin)
            st.session_state.intermediates.append(("Skeleton", current.copy()))
        elif op == "Convex Hull":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = convex_hull_bin(cur_bin)
            st.session_state.intermediates.append(("ConvexHull", current.copy()))
        elif op == "Top Hat (contrast)":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_tophat(g, kernel_size)
            st.session_state.intermediates.append(("TopHat", current.copy()))
        elif op == "Black Hat (contrast)":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = op_blackhat(g, kernel_size)
            st.session_state.intermediates.append(("BlackHat", current.copy()))
        elif op == "Beucher Gradient":
            cur_bin = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = beucher_gradient(cur_bin, kernel_size)
            st.session_state.intermediates.append(("BeucherGrad", current.copy()))
        elif op == "Morphological Laplace":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = morphological_laplace(g, kernel_size)
            st.session_state.intermediates.append(("MorphLaplace", current.copy()))
        elif op == "Local Entropy Map":
            g = current if current.ndim==2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            current = local_entropy_map(g, entropy_radius)
            st.session_state.intermediates.append(("LocalEntropy", current.copy()))

    st.session_state.result = current

# --------------------------
# Display results
# --------------------------
st.subheader("📊 Kết quả")

col_a, col_b = st.columns([1,1])
with col_a:
    st.write("**Original**")
    st.image(cv_to_pil(img_cv), use_container_width=True)

with col_b:
    st.write("**Processed result**")
    if st.session_state.result is not None:
        pil_res = cv_to_pil(st.session_state.result) if isinstance(st.session_state.result, np.ndarray) else Image.fromarray(st.session_state.result)
        st.image(pil_res, use_container_width=True)
    else:
        st.info("Nhấn 'Run Pipeline' để xem kết quả.")

# Comparison blend slider
if st.session_state.result is not None:
    st.markdown("### 🔀 Comparison blend (interactive)")
    alpha = blend_alpha
    orig_pil = cv_to_pil(img_cv)
    res_pil = cv_to_pil(st.session_state.result).convert("RGB")
    res_pil = res_pil.resize(orig_pil.size)
    blended = Image.blend(orig_pil.convert("RGB"), res_pil, alpha=alpha)
    st.image(blended, use_container_width=True)

# Show pipeline steps if requested
if show_pipeline and len(st.session_state.intermediates) > 0:
    st.markdown("### 🧭 Intermediate steps")
    n = len(st.session_state.intermediates)
    cols = st.columns(min(4, n))
    for i, (name, im) in enumerate(st.session_state.intermediates):
        col = cols[i % len(cols)]
        col.write(name)
        col.image(cv_to_pil(im), use_container_width=True)

# Download result
if st.session_state.result is not None:
    buf = BytesIO()
    pil_res = cv_to_pil(st.session_state.result)
    pil_res.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("💾 Download processed image", data=byte_im, file_name="morph_result.png", mime="image/png")

# Small explanation panel
with st.expander("ℹ️ Giải thích ngắn mỗi bước"):
    st.markdown("""
    - **Skeletonize**: rút xương (thin) của đối tượng — tốt cho vân tay/ OCR.
    - **Convex Hull**: tìm bao lồi của đối tượng.
    - **Remove Small Objects**: loại bỏ các connected components nhỏ theo kích thước.
    - **Extract Boundary / Beucher Gradient**: lấy biên bằng dilation - erosion.
    - **Top Hat / Black Hat**: tăng cường chi tiết sáng / tối nhỏ (morphological contrast).
    - **Local Entropy Map**: đo độ phức tạp cục bộ (texture).
    - **Morphological Laplace**: kết hợp gradient morphology + Laplacian để tăng chi tiết.
    """)