import os
import json
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

class NoiseRobustSkinModel(nn.Module):
    def __init__(self, backbone_name, num_classes=7, dropout=0.5, pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )

        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            in_features = self.backbone.get_classifier().in_features

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        out = self.head(feats)
        if return_features:
            return out, feats
        return out

# =============================
# Page configuration
# =============================
st.set_page_config(
    page_title="RefineMed | Research Prototype",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Styling
# =============================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 1.6rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 46px;
        border: none;
        font-weight: 600;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    .muted {
        color: #6b7280;
        font-size: 0.95rem;
    }
    .small-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #374151;
        margin-bottom: 8px;
    }
    .pred-label {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }
    .pred-score {
        font-size: 2rem;
        font-weight: 800;
        color: #0f766e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Configuration
# =============================
NUM_CLASSES = 7
IMAGE_SIZE = 224
DEFAULT_BACKBONE = os.getenv("MODEL_BACKBONE", "efficientnet_b0")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "rehab_adaptive_idn20_best.pth"))
DEFAULT_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", os.path.join(MODELS_DIR, "model_config.json"))

CLASS_NAMES = {
    0: "Actinic Keratosis (akiec)",
    1: "Basal Cell Carcinoma (bcc)",
    2: "Benign Keratosis (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic Nevus (nv)",
    6: "Vascular Lesion (vasc)",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================
# Utility functions
# =============================
def load_optional_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_model(backbone: str, num_classes: int, dropout: float = 0.5):
    return NoiseRobustSkinModel(
        backbone_name=backbone,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False
    )


@st.cache_resource
def load_model(model_path: str, config_path: str):
    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"

    device = torch.device("cpu")
    cfg = load_optional_config(config_path)

    backbone = cfg.get("backbone", "efficientnet_b0")
    dropout = float(cfg.get("dropout", 0.5))
    num_classes = int(cfg.get("num_classes", 7))

    model = build_model(backbone, num_classes, dropout)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    model.to(device)

    if missing or unexpected:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()

    metadata = {
        "backbone": backbone,
        "dropout": dropout,
        "num_classes": num_classes,
        "device": str(device),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }

    return model, metadata, None


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_image(model: nn.Module, image: Image.Image) -> Tuple[int, float, torch.Tensor, float]:
    transform = get_inference_transform()
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)

    top_prob, top_idx = probs.max(dim=1)
    return int(top_idx.item()), float(top_prob.item()), probs.squeeze(0), float(entropy.item())


def confidence_band(conf: float) -> Tuple[str, str]:
    if conf >= 0.80:
        return "High confidence", "The model output is relatively confident for this image."
    if conf >= 0.60:
        return "Moderate confidence", "The model output is moderately confident, but alternative classes may still be relevant."
    return "Low confidence", "This image may be ambiguous, difficult, or outside the model's strongest decision boundary."


def entropy_band(entropy_value: float, num_classes: int = NUM_CLASSES) -> Tuple[str, str]:
    max_entropy = torch.log(torch.tensor(float(num_classes))).item()
    ratio = entropy_value / max_entropy if max_entropy > 0 else 0.0

    if ratio < 0.35:
        return "Low ambiguity", "The probability distribution is fairly concentrated around one class."
    if ratio < 0.60:
        return "Moderate ambiguity", "The image shows some uncertainty across classes."
    return "High ambiguity", "The model is uncertain and multiple classes receive meaningful probability mass."


def format_topk(probabilities: torch.Tensor, k: int = 3) -> List[Tuple[str, float]]:
    top_probs, top_indices = torch.topk(probabilities, k=k)
    rows = []
    for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
        rows.append((CLASS_NAMES.get(int(idx), f"Class {idx}"), float(p)))
    return rows


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.title("RefineMed")
    st.caption("Research prototype")

    st.markdown("---")
    st.markdown(
        """
        **Project focus**
        - Skin lesion image classification
        - Robust training under noisy annotations
        - Confidence-aware research demonstration
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **Training context**
        - Dataset: HAM10000
        - Backbone: EfficientNet-B0
        - Training idea: SIEVE + REHAB
        - Evaluation metrics: Accuracy, Macro-F1, MCC
        """
    )

    st.markdown("---")
    st.warning(
        "This interface is for academic research demonstration only. It is not a diagnostic or clinical decision tool."
    )


# =============================
# Main layout
# =============================
st.title("🔬 RefineMed: Robust Skin Lesion Classification")
st.markdown(
    "Demonstration interface for the final trained model. The noisy-annotation handling happens during training, while this UI shows the resulting prediction behavior, confidence, and ambiguity at inference time."
)

model, metadata, load_error = load_model(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH)

if load_error:
    st.error(load_error)
    st.info("Update MODEL_PATH and MODEL_CONFIG_PATH, or place the files inside the models folder.")
    st.stop()

if metadata["missing_keys"] or metadata["unexpected_keys"]:
    st.warning("Checkpoint loaded with key mismatches. Please verify compatibility.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Input Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            image = None
    else:
        st.info("Upload a dermoscopic image to run the research prototype.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Model Information</div>', unsafe_allow_html=True)
    st.write(f"**Backbone:** {metadata['backbone']}")
    st.write(f"**Dropout:** {metadata['dropout']}")
    st.write("**Inference mode:** CPU")
    st.write("**Class set:** akiec, bcc, bkl, df, mel, nv, vasc")
    st.write(f"**Checkpoint:** {os.path.basename(DEFAULT_MODEL_PATH)}")
    st.write(f"**Missing keys:** {metadata['missing_keys']}")
    st.write(f"**Unexpected keys:** {metadata['unexpected_keys']}")
    st.markdown('<div class="muted">The UI loads a saved checkpoint from the training pipeline and performs inference only.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Prediction Summary</div>', unsafe_allow_html=True)

    if image is not None:
        analyze = st.button("Analyze Image")
    else:
        analyze = False

    if analyze and image is not None:
        top_idx, top_conf, probs, entropy_value = predict_image(model, image)
        pred_label = CLASS_NAMES.get(top_idx, f"Class {top_idx}")
        conf_text, conf_note = confidence_band(top_conf)
        ambiguity_text, ambiguity_note = entropy_band(entropy_value)
        st.write(f"**Predicted class index:** {top_idx}")
        st.write(f"**Raw probabilities:** {[round(float(p), 6) for p in probs.tolist()]}")

        st.markdown(f'<div class="pred-label">{pred_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pred-score">{top_conf * 100:.1f}%</div>', unsafe_allow_html=True)
        st.progress(min(max(float(top_conf), 0.0), 1.0))

        st.write(f"**Confidence category:** {conf_text}")
        st.write(conf_note)
        st.write(f"**Ambiguity estimate:** {ambiguity_text}")
        st.write(ambiguity_note)

        topk_rows = format_topk(probs, k=3)
        st.markdown("### Top-3 Predictions")
        for label, p in topk_rows:
            st.write(f"- **{label}**: {p * 100:.1f}%")

        st.markdown("### Interpretation Notes")
        if top_conf < 0.60:
            st.warning(
                "Low-confidence output. The image may be difficult, ambiguous, or visually similar to other classes."
            )
        elif top_conf < 0.80:
            st.info(
                "Moderate-confidence output. Alternative classes should still be considered when interpreting the result."
            )
        else:
            st.success(
                "The model is relatively confident in this prediction for the uploaded image."
            )

        st.markdown("### Research Disclaimer")
        st.caption(
            "This result is generated by an academic research prototype trained on HAM10000. It should not be used for diagnosis, treatment, or clinical decision-making."
        )
    else:
        st.info("Upload an image and click **Analyze Image** to view the model output.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    **How this relates to the project**

    - The uploaded image is passed through the final trained classification model.
    - The noisy-annotation handling does not happen inside the UI. It was performed during model training through the SIEVE + REHAB pipeline.
    - This interface demonstrates the final inference behavior, including class probabilities, confidence, and ambiguity-aware output.
    """
)
