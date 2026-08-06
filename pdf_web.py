import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="ScanWala Live", layout="centered", page_icon="📄")

# ====== YE NAYI LINE ADD KI HAI - GITHUB BUTTON CHUPANE KE LIYE ======
st.markdown("""
<style>
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)
# ====================================================================

# DARK BLUE THEME + CENTERED
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

.stApp {
    background: linear-gradient(135deg,#0a0f1a 0%,#101c2e 50%,#182a42 100%);
}

/* Niche se Upar 50 Particles */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        radial-gradient(circle, rgba(30,144,255,1) 2px, transparent 2px),
        radial-gradient(circle, rgba(100,180,255,1) 1.5px, transparent 1.5px),
        radial-gradient(circle, rgba(0,191,255,1) 2.5px, transparent 2.5px);
    background-size: 50px 50px, 70px 70px, 90px 90px;
    background-position: 0 100vh, 0 100vh, 0 100vh;
    animation: riseUp 6s linear infinite;
}

@keyframes riseUp {
    0% { background-position: 0 100vh, 0 100vh, 0 100vh; }
    100% { background-position: 0 -100vh, 0 -100vh, 0 -100vh; }
}

.block-container {
    background: rgba(10, 15, 25, 0.85);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    border: 1px solid rgba(30, 144, 255, 0.3);
    box-shadow: 0 0 40px rgba(30, 144, 255, 0.4);
    margin: auto;
    max-width: 700px;
}

h1 {
    font-family: 'Poppins', sans-serif;
    color: white!important;
    text-align: center;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(30, 144, 255, 0.8);
}

.stButton>button {
    background: linear-gradient(90deg, #1E90FF, #00BFFF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>📄Live Scanner</h1>", unsafe_allow_html=True)
st.markdown('<p style="color:rgba(255,255,255,0.8); text-align:center;">Upload documents with live scanning</p>', unsafe_allow_html=True)

# ========== IMAGE PROCESSING CODE ==========
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def find_document(img):
    h, w = img.shape[:2]
    area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area < area * 0.15:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx)!= 4:
            continue
        x, y, cw, ch = cv2.boundingRect(approx)
        if cw < 200 or ch < 200:
            continue
        ratio = cw / float(ch)
        if ratio < 0.5 or ratio > 0.9:
            continue
        if c_area > best_area:
            best_area = c_area
            best = approx
    return best

def process_image(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(gray_rgb)

def images_to_pdf(images_pil, output_path):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(False, margin=0)
    for img in images_pil:
        temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp.close()
        try:
            img.save(temp.name, "JPEG", quality=95)
            pdf.add_page()
            pdf.image(temp.name, x=0, y=0, w=210, h=297)
        finally:
            os.remove(temp.name)
    pdf.output(output_path)
# ========== END PROCESSING CODE ==========

# ========== STREAMLIT UI ==========
uploaded_files = st.file_uploader(
    "Choose images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key="main_uploader"
)

if uploaded_files:
    processed_images = []
    st.markdown("### 📸 Processed Preview")

    for file in uploaded_files:
        file_bytes = file.getvalue()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        processed = process_image(img)

        if processed is not None:
            processed_images.append(processed)
            st.image(processed, caption=f"Processed: {file.name}", use_container_width=True)
            st.divider()

    if processed_images and st.button("🚀 Generate PDF", type="primary"):
        with st.spinner("Creating PDF..."):
            os.makedirs("output", exist_ok=True)
            output_file = "output/ScanWala.pdf"
            images_to_pdf(processed_images, output_file)

        st.success("✅ PDF Ready!")
        with open(output_file, "rb") as f:
            st.download_button("📥 Download PDF", f, "ScanWala.pdf", "application/pdf")

else:
    st.info("Please upload 1 or more images to start")
