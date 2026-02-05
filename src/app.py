import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import os

#page configuration
st.set_page_config(
    page_title="RefineMed | Skin Lesion Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    .metric-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

#configuration
NUM_CLASSES = 7

#finds experiments folder relative to this script file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#point to the refinement model
# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "data",
#     "experiments",
#     "Phase8_Refinement",
#     "new_best_model_refinement.pth"
# )

PHASE8_DIR = os.path.join(BASE_DIR, "models", "phase8")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(PHASE8_DIR, "new_best_model_refinement.pth")
)


#this mapping must match training code exactly!
#0:nv, 1:mel, 2:bkl, 3:bcc, 4:akiec, 5:vasc, 6:df
CLASSES = {
    0: 'Melanocytic Nevus (Mole)',
    1: 'Melanoma (High Risk)',
    2: 'Benign Keratosis',
    3: 'Basal Cell Carcinoma',
    4: 'Actinic Keratosis',
    5: 'Vascular Lesion',
    6: 'Dermatofibroma'
}

#load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file missing!")
        st.code(f"Looked for:\n{MODEL_PATH}")
        st.info("Place the .pth file in the Phase8_Refinement folder or set an environment variable MODEL_PATH.")
        return None

    device = torch.device("cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.eval()
    return model

#sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("RefineMed Prototype")
    
    st.markdown("---")
    st.write("This tool identifies 7 common skin lesions. It uses a **Sieve & Rehabilitate** framework to handle ambiguous (noisy) medical images.")
    
    st.markdown("### Performance")
    #METRICS
    st.write("✅ **Accuracy (Val):** ~80.8%")
    st.write("✅ **MCC (Best):** 0.6214")
    st.write("✅ **Macro-F1:** ~0.568")
    st.markdown("---")
    st.caption("⚠️ For Research/Academic Use Only.")

#main layout
st.title("🔍 AI-Powered Skin Lesion Analysis")
st.markdown("### Clinical Support System")
st.write("Upload a dermoscopic image below for instant classification.")

col1, col2 = st.columns([1, 1], gap="large")

# --- COLUMN 1: UPLOAD ---
with col1:
    st.subheader("1. Patient Skin Image")
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Scan", use_container_width=True)

# --- COLUMN 2: RESULTS ---
with col2:
    st.subheader("2. Diagnostic Report")
    
    if uploaded_file:
        # Load model only when needed
        model = load_model()
        
        if st.button("🔍 Analyze Lesion"):
            if model:
                # Progress bar effect
                progress_text = "Processing image..."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                my_bar.empty()

                # Inference Transforms (Must match training)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_probs, top_classes = probs.topk(3, dim=1)

                    # Primary prediction = top-1
                    confidence = float(top_probs[0][0]) * 100
                    class_idx = int(top_classes[0][0])
                    pred_label = CLASSES.get(class_idx, "Unknown")

                    st.write("Top predictions:")
                    for p, c in zip(top_probs[0], top_classes[0]):
                        st.write(f"- {CLASSES.get(int(c), 'Unknown')}: {float(p)*100:.1f}%")

                    if confidence < 50:
                        st.warning("⚠️ Low confidence prediction. Consider a clinical review or better image quality.")


                # --- DISPLAY RESULTS ---
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #555; margin-bottom: 5px;">Primary Detection</h3>
                    <h2 style="color: #2c3e50; font-size: 28px; margin: 10px 0;">{pred_label}</h2>
                    <h1 style="color: #27ae60; font-size: 48px; margin: 0;">{confidence:.1f}%</h1>
                    <p style="color: #888;">Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # --- CLINICAL ALERTS ---
                # 1 = Melanoma, 3 = BCC, 4 = AKIEC (Pre-cancerous)
                if confidence < 65:
                    st.warning("⚠️ Uncertain prediction. Please use a clearer dermoscopic image and seek clinical review.")
                elif class_idx in [1, 3]:
                    st.error("🚨 HIGH RISK ALERT: Possible malignancy. Immediate dermatological referral is recommended.")
                elif class_idx == 4:
                    st.warning("⚠️ CAUTION: Possible pre-cancerous growth (Actinic Keratosis). Clinical follow-up advised.")
                else:
                    st.success("✅ LOW RISK: The lesion appears benign. Standard monitoring protocols apply.")

    else:
        # Placeholder State
        st.info("👈 Please upload an image to begin analysis.")
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; color: #ccc;">
            <h3>Waiting for Image...</h3>
            <p>Select a file from the panel on the left.</p>
        </div>
        """, unsafe_allow_html=True)