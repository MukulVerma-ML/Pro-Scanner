import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="ScanWala Live", layout="centered", page_icon="📄")

# FIXED LIVE BACKGROUND
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #1a0f0a 0%, #3d1f10 50%, #5c2f15 100%);
}

/* Background glow layers */
.stApp {
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(255, 140, 0, 0.25) 0%, transparent 40%),
        radial-gradient(circle at 80% 20%, rgba(255, 200, 100, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 40% 80%, rgba(255, 100, 0, 0.15) 0%, transparent 40%),
        linear-gradient(135deg, #1a0f0a 0%, #3d1f10 50%, #5c2f15 100%);
    animation: bgMove 20s ease-in-out infinite;
}

@keyframes bgMove {
    0%, 100% { background-position: 0% 0%, 100% 0%, 0% 100%, 0% 0%; }
    50% { background-position: 10% 10%, 90% 10%, 10% 90%, 0% 0%; }
}

/* Glass card */
.block-container {
    background: rgba(20, 20, 20, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    border: 1px solid rgba(255, 140, 0, 0.2);
    box-shadow: 0 0 40px rgba(255, 140, 0, 0.3);
    margin-top: 2rem;
}

h1 {
    font-family: 'Poppins', sans-serif;
    color: white !important;
    text-align: center;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(255, 140, 0, 0.8);
}

.stButton>button {
    background: linear-gradient(90deg, #FF8C00, #FFA500);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(255, 140, 0, 0.5);
}

/* Fake particles using background */
[data-testid="stAppViewContainer"] {
    background-image: 
        radial-gradient(2px 2px at 50px 100px, rgba(255,180,0,0.8), transparent),
        radial-gradient(3px 3px at 150px 200px, rgba(255,140,0,0.6), transparent),
        radial-gradient(2px 2px at 250px 50px, rgba(255,200,100,0.7), transparent),
        radial-gradient(2px 2px at 350px 150px, rgba(255,180,0,0.8), transparent),
        radial-gradient(3px 3px at 100px 300px, rgba(255,140,0,0.6), transparent);
    background-size: 400px 400px;
    animation: twinkle 6s linear infinite;
}

@keyframes twinkle {
    0% { background-position: 0px 0px; }
    100% { background-position: 0px -400px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📄Live Scanner</h1>", unsafe_allow_html=True)
st.markdown('<p style="color:rgba(255,255,255,0.8); text-align:center;">Upload documents with live scanning</p>', unsafe_allow_html=True)
st.title("📄Image to PDF")
st.write("Upload multiple images. The app will auto-crop, straighten, and convert them to 1 PDF.")

# ========== YOUR ORIGINAL CODE START ==========
# Order 4 corner points
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)] # TR
    rect[3] = pts[np.argmax(d)] # BL
    return rect

# Find reliable document contour
def find_document(img):
    h, w = img.shape[:2]
    area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for c in contours:
        ratio = cv2.contourArea(c) / area
        if ratio < 0.30:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx)!= 4:
            continue
        pts = approx.reshape(4, 2)
        x, y, cw, ch = cv2.boundingRect(approx)
        if (cw * ch) / area < 0.35:
            continue
        mx, my = w * 0.05, h * 0.05
        near_edges = sum([
            np.min(pts[:, 0]) < mx,
            np.max(pts[:, 0]) > w - mx,
            np.min(pts[:, 1]) < my,
            np.max(pts[:, 1]) > h - my
        ])
        score = ratio + (0.30 if near_edges >= 3 else 0)
        if score > best_score:
            best_score = score
            best = approx
    return best if best_score >= 0.50 else None

# Process one image - Modified to take numpy array instead of path
def process_image(img):
    if img is None:
        return None

    # Rotate landscape image
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Detect document
    contour = find_document(img)
    if contour is not None:
        pts = order_points(contour.reshape(4, 2).astype("float32"))
        tl, tr, br, bl = pts
        W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        ratio = W / H
        if W >= 200 and H >= 200 and 0.3 < ratio < 3:
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            img = cv2.warpPerspective(img, M, (W, H))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bright documents -> B&W
    if np.mean(gray) > 140:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    return Image.fromarray(gray).convert("RGB")

# Convert images to PDF
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
# ========== YOUR ORIGINAL CODE END ==========

# ========== STREAMLIT UI - SIRF 1 BAAR ==========
uploaded_files = st.file_uploader(
    "Choose images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key="main_uploader" # unique key add kar diya
)

if uploaded_files:
    processed_images = []
    col1, col2 = st.columns(2)

    for i, file in enumerate(uploaded_files):
        # Convert uploaded file to cv2 image
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        processed = process_image(img)
        if processed:
            processed_images.append(processed)
            with col1 if i % 2 == 0 else col2:
                st.image(processed, caption=f"Processed: {file.name}", use_column_width=True)

    if processed_images and st.button("🚀 Generate PDF", type="primary"):
        with st.spinner("Creating PDF..."):
            os.makedirs("output", exist_ok=True)
            output_file = "output/opencv_fixed.pdf"
            images_to_pdf(processed_images, output_file)

        st.success(f"Done! PDF saved")
        with open(output_file, "rb") as f:
            st.download_button(
                label="📥 Download PDF",
                data=f,
                file_name="ScanWala.pdf",
                mime="application/pdf"
            )
else:
    st.info("Please upload 1 or more images to start")