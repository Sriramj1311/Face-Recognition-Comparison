"""Comparative Face Detection & Recognition - Streamlit application."""
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from modules import facenet_module, deepface_module, template_matching, viola_jones
from content import METHODS, COMPARISON, VIVA

BASE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE, "database", "reference_faces")
RESULT_DIR = os.path.join(BASE, "results")
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

st.set_page_config(
    page_title="VisionLab | Face Recognition",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {padding: 1.8rem 2rem; border-radius: 20px; background: linear-gradient(135deg,#111827,#243b53);
        color:white; margin-bottom:1.5rem; border:1px solid #334155;}
.hero h1 {font-size:2.35rem; margin:0 0 .4rem 0;}
.hero p {color:#cbd5e1; margin:0;}
.badge {display:inline-block; padding:.3rem .65rem; border-radius:999px; background:#1e293b;
        border:1px solid #475569; margin:.2rem .25rem .2rem 0; font-size:.82rem;}
.section-title {font-size:1.35rem; font-weight:700; margin-top:1rem;}
[data-testid="stMetric"] {border:1px solid #e2e8f0; padding:10px; border-radius:12px;}
div[data-testid="stFileUploader"] {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

def decode(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def safe_run(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

def load_references():
    refs = []
    for f in sorted(os.listdir(REF_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(REF_DIR, f))
            if img is not None:
                refs.append((f, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return refs

def save_output(name, result):
    if result.get("ok") and result.get("image") is not None:
        path = os.path.join(RESULT_DIR, f"{name}_result.png")
        cv2.imwrite(path, cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))

with st.sidebar:
    st.markdown("## ⚙️ Project Controls")
    st.caption("Configure reference images and template matching.")
    ref_files = st.file_uploader(
        "Reference face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Known-person images used by FaceNet and DeepFace.",
    )
    if ref_files and st.button("💾 Save references", use_container_width=True):
        for rf in ref_files:
            with open(os.path.join(REF_DIR, os.path.basename(rf.name)), "wb") as fh:
                fh.write(rf.getvalue())
        st.success(f"Saved {len(ref_files)} image(s).")

    tmpl_file = st.file_uploader(
        "Template image",
        type=["jpg", "jpeg", "png"],
        key="tmpl",
        help="Optional image patch for classical template matching.",
    )
    st.divider()
    references = load_references()
    uploaded_names = {n for n, _ in references}
    for rf in ref_files or []:
        img = decode(rf)
        if img is not None and rf.name not in uploaded_names:
            references.append((rf.name, img))
    st.metric("Reference faces in use", len(references))
    st.caption("Tip: clear, front-facing reference images give better recognition results.")

st.markdown("""
<div class="hero">
  <h1>🔎 VisionLab — Face Detection & Recognition</h1>
  <p>Compare four computer-vision approaches on the same image: FaceNet, DeepFace, Template Matching and Viola-Jones.</p>
  <div>
    <span class="badge">Deep Learning</span><span class="badge">OpenCV</span>
    <span class="badge">Face Recognition</span><span class="badge">Computer Vision</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_main, tab_about, tab_viva = st.tabs(["🧪 Analysis", "📚 Methods", "🎓 Viva Questions"])

with tab_main:
    uploaded = st.file_uploader(
        "Upload an image to analyze",
        type=["jpg", "jpeg", "png"],
        help="The same image is independently processed by all four methods.",
    )
    if uploaded is None:
        st.info("👆 Upload a JPG or PNG image above to start the comparison.")
        st.markdown("### What this demo compares")
        cols = st.columns(4)
        for col, (name, desc) in zip(cols, [
            ("FaceNet","Embedding-based recognition"),
            ("DeepFace","Verification & facial analysis"),
            ("Template Matching","Classical image pattern matching"),
            ("Viola-Jones","Classical face detection"),
        ]):
            col.markdown(f"**{name}**")
            col.caption(desc)
    else:
        image = decode(uploaded)
        if image is None:
            st.error("Invalid image file. Please upload a valid JPG or PNG.")
        else:
            st.markdown("### Original image")
            st.image(image, width=520)

            template = decode(tmpl_file) if tmpl_file else None
            with st.spinner("Running the four methods. First use may download pretrained models..."):
                r_fn = safe_run(facenet_module.run, image, references)
                r_df = safe_run(deepface_module.run, image, references)
                r_tm = safe_run(template_matching.run, image, template)
                r_vj = safe_run(viola_jones.run, image)

            st.markdown("### Results")
            result_tabs = st.tabs(["1 · FaceNet", "2 · DeepFace", "3 · Template Matching", "4 · Viola-Jones"])

            with result_tabs[0]:
                if not r_fn["ok"]:
                    st.error(r_fn["error"])
                else:
                    c1,c2 = st.columns([2,1])
                    c1.image(r_fn["image"], caption="Detected face")
                    c2.image(r_fn["face"], caption="Cropped face", width=240)
                    st.success(r_fn["message"])
                    e = r_fn["embedding"]
                    st.metric("Embedding dimensions", e.shape[0])
                    st.caption(f"First 8 values: {np.round(e[:8], 3).tolist()}")
                    if r_fn["matches"]:
                        st.dataframe(pd.DataFrame(r_fn["matches"]), use_container_width=True, hide_index=True)
                save_output("facenet", r_fn)

            with result_tabs[1]:
                if not r_df["ok"]:
                    st.error(r_df["error"])
                else:
                    c1,c2 = st.columns([2,1])
                    c1.image(r_df["image"], caption="Detected face")
                    c2.image(r_df["face"], caption="Cropped face", width=240)
                    st.success(r_df["message"])
                    st.metric("Recognition model", r_df["model"])
                    if r_df["matches"]:
                        st.dataframe(pd.DataFrame(r_df["matches"]), use_container_width=True, hide_index=True)
                    st.caption("Attribute estimates, when returned by the model, are approximate.")
                    st.json(r_df["attributes"])
                save_output("deepface", r_df)

            with result_tabs[2]:
                if not r_tm["ok"]:
                    st.error(r_tm["error"])
                else:
                    c1,c2 = st.columns([1,2])
                    c1.image(r_tm["template"], caption="Template", width=240)
                    c2.image(r_tm["image"], caption="Best matching area")
                    m1,m2,m3 = st.columns(3)
                    m1.metric("Score", f"{r_tm['score']:.3f}")
                    m2.metric("Found", "Yes" if r_tm["found"] else "No")
                    m3.metric("Top-left", str(r_tm["location"]))
                    (st.success if r_tm["found"] else st.warning)(r_tm["message"])
                    st.caption("Template matching compares image patterns; it does not identify a person.")
                save_output("template", r_tm)

            with result_tabs[3]:
                if not r_vj["ok"]:
                    st.error(r_vj["error"])
                else:
                    st.image(r_vj["image"], caption="Haar Cascade detections")
                    st.metric("Faces detected", r_vj["count"])
                    st.write("Bounding boxes (x, y, width, height):", r_vj["boxes"])
                save_output("viola_jones", r_vj)

            st.markdown("### 📊 Side-by-side comparison")
            rows = []
            for name, r in [("FaceNet",r_fn),("DeepFace",r_df),("Template Matching",r_tm),("Viola-Jones",r_vj)]:
                t,p,a,l = COMPARISON[name]
                rows.append({"Method":name,"Type":t,"Main Purpose":p,
                             "Result":r.get("summary", "FAILED: "+r.get("error","unknown")) if r.get("ok") else "FAILED: "+r["error"],
                             "Advantages":a,"Limitations":l})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_about:
    st.markdown("### How the four approaches differ")
    for name, m in METHODS.items():
        with st.expander(name, expanded=False):
            st.markdown(f"**What it is:** {m['what']}")
            st.markdown(f"**How it works:** {m['how']}")
            st.markdown(f"**Input:** {m['input']}  \n**Output:** {m['output']}")
            st.markdown(f"**Advantages:** {m['adv']}")
            st.markdown(f"**Limitations:** {m['lim']}")
            st.markdown(f"**Applications:** {m['apps']}")

with tab_viva:
    st.markdown("### 🎓 Viva preparation")
    for i, (q,a) in enumerate(VIVA,1):
        with st.expander(f"Q{i}. {q}"):
            st.write(a)
