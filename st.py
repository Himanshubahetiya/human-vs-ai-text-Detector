import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Human & AI Detector",
    page_icon="🔍",
    layout="wide"
)

# --- THE ULTIMATE NEON GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background: #020617;
        background-image: 
            radial-gradient(at 0% 0%, rgba(112, 0, 255, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(0, 242, 255, 0.12) 0px, transparent 50%);
        color: #f8fafc;
    }

    /* --- CUSTOM NAVBAR --- */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 5%;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(15px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
    }

    /* --- HERO TEXT --- */
    .glitch-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #fff 20%, #7000ff 50%, #00f2ff 80%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        text-transform: uppercase;
        margin-bottom: 0;
    }
    @keyframes shine { to { background-position: 200% center; } }

    /* --- GLASS CARDS --- */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* --- IMPROVED INPUT AREA --- */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(112, 0, 255, 0.4) !important;
        border-radius: 20px !important;
        color: #00f2ff !important;
        font-size: 1.1rem !important;
        padding: 20px !important;
        transition: 0.3s all !important;
    }
    .stTextArea textarea:focus {
        border-color: #00f2ff !important;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2) !important;
    }

    /* --- BUTTONS --- */
    div.stButton > button {
        background: #fff !important;
        color: #000 !important;
        border-radius: 50px !important;
        padding: 10px 30px !important;
        font-weight: 700 !important;
        transition: 0.3s all !important;
        border: none !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 242, 255, 0.4) !important;
        background: #00f2ff !important;
    }

    .counter-label {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }

    header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- FEATURE EXTRACTION LOGIC ---
def extract_features(text):
    words = text.split()
    word_count = len(words)
    # Basic counts
    sentence_count = text.count(".") + text.count("?") + text.count("!")
    if sentence_count == 0: sentence_count = 1
    
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_len = word_count / sentence_count
    unique_ratio = len(set(words)) / word_count if word_count else 0
    
    # Stylometry
    punctuation_count = sum(text.count(p) for p in ".,!?;:")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    digit_count = sum(c.isdigit() for c in text)
    stopword_ratio = sum(1 for w in words if w.lower() in ENGLISH_STOP_WORDS) / word_count if word_count else 0
    avg_char_sentence = len(text) / sentence_count
    
    # Burstiness & Variation
    sentences = text.split(".")
    lengths = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    burstiness = np.std(lengths) if lengths else 0
    repetition = 1 - (len(set(words)) / word_count) if word_count else 0

    return [
        word_count, sentence_count, avg_word_len, avg_sentence_len,
        unique_ratio, punctuation_count, uppercase_ratio, digit_count,
        stopword_ratio, avg_char_sentence, burstiness, repetition
    ]

# --- ASSET LOADING ---
@st.cache_resource
def load_engine():
    try:
        with open("text.pkl", "rb") as f:
            data = pickle.load(f)
        return data["model"], data["tfidf"], data["scaler"]
    except:
        return None, None, None

model, tfidf, scaler = load_engine()

# --- STATE MANAGEMENT ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def set_page(page):
    st.session_state.current_page = page

# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
    <div class="nav-bar">
        <div style="font-size: 1.4rem; font-weight: 700; color: #fff;">DETECTOR<span style="color:#7000ff;">.AI</span></div>
        <div style="color: #64748b; font-size: 0.85rem;">Neural Forensic Engine • v2.0</div>
    </div>
""", unsafe_allow_html=True)
st.write("<br><br><br><br>", unsafe_allow_html=True)

# ==========================================
# PAGE: HOME
# ==========================================
if st.session_state.current_page == 'home':
    c1, c2 = st.columns([8, 2])
    with c2: st.button("Launch App 🚀", on_click=lambda: set_page('app'))

    st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h1 class="glitch-title">Human and AI<br>Text Detection</h1>
            <p style="color: #94a3b8; font-size: 1.2rem; max-width: 650px; margin: 15px auto;">
                Advanced neural forensic analysis to distinguish between 
                synthetic and organic manuscripts using SVM & Stylometry.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass-card"><h3 style="color:#00f2ff; margin-top:0;">Identify AI</h3><p style="color:#94a3b8; font-size:0.9rem;">Scan for algorithmic patterns, low burstiness, and robotic consistency.</p></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="glass-card"><h3 style="color:#7000ff; margin-top:0;">Verify Human</h3><p style="color:#94a3b8; font-size:0.9rem;">Confirm creative variance, linguistic complexity, and natural flow.</p></div>', unsafe_allow_html=True)

# ==========================================
# PAGE: APP (ANALYSIS CONSOLE)
# ==========================================
else:
    c1, c2 = st.columns([8, 2])
    with c2: st.button("Close App ✖", on_click=lambda: set_page('home'))

    st.write("<br>", unsafe_allow_html=True)
    l_col, r_col = st.columns([7, 3])
    
    with l_col:
        st.markdown("<h2 style='margin-bottom:5px; color:#fff;'>Analysis Console</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; margin-bottom:20px;'>Input text below for forensic linguistic verification.</p>", unsafe_allow_html=True)
        
        user_input = st.text_area(
            "input", 
            placeholder="Paste text here (minimum 50 words recommended)...", 
            height=350, 
            label_visibility="collapsed"
        )
        
        # Counters
        w_count = len(user_input.split()) if user_input.strip() else 0
        c_count = len(user_input)
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-top: -15px; padding: 10px 5px;">
                <div class="counter-label">WORDS: <span style="color:#00f2ff;">{w_count}</span></div>
                <div class="counter-label">CHARS: <span style="color:#7000ff;">{c_count}</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚡ SCAN FOR PATTERNS"):
            if not user_input.strip():
                st.toast("Input required.", icon="⚠️")
            elif model and tfidf and scaler:
                with st.spinner("Decoding neural signatures..."):
                    time.sleep(1.2)

                    # Processing
                    text_vector = tfidf.transform([user_input])
                    features = extract_features(user_input)
                    style_feature = scaler.transform([features])
                    
                    # Merge and Predict
                    final_vector = hstack([text_vector, style_feature])
                    prediction = model.predict(final_vector)[0]
                    confidence_score = model.decision_function(final_vector)[0]

                    if prediction == 0:
                        st.markdown(
                            f'<div style="margin-top:20px; padding:30px; border-radius:20px; background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; text-align:center;"><h2 style="color:#10b981; margin:0;">✅ HUMAN WRITTEN</h2><p style="color:#f8fafc; opacity:0.8; margin-top:10px;"></p></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div style="margin-top:20px; padding:30px; border-radius:20px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; text-align:center;"><h2 style="color:#ef4444; margin:0;">⚠️ AI GENERATED</h2><p style="color:#f8fafc; opacity:0.8; margin-top:10px;"></p></div>',
                            unsafe_allow_html=True
                        )
            else:
                st.error("Engine files (text.pkl) missing or corrupted.")

    with r_col:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#fff; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;">System Status</h4>
                <div style="margin-top:15px;">
                    <p style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">Scanner: <span style="color:#10b981; float:right;">● Active</span></p>
                    <p style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">Stylometry: <span style="color:#00f2ff; float:right;">Enabled</span></p>
                    <p style="font-size:0.85rem; color:#94a3b8;">Engine: <span style="color:#7000ff; float:right;">SVM + TF-IDF + Scaler</span></p>
                </div>
            </div>
            <div style="margin-top:20px; padding:15px; background:rgba(0,242,255,0.05); border-radius:15px; border: 1px dashed rgba(0,242,255,0.2);">
                <p style="font-size:0.75rem; color:#00f2ff; margin:0; line-height:1.4;">
                    <b>NOTE:</b> This engine analyzes linguistic variance (burstiness) and stopword distribution to detect synthetic patterns.
                </p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br><center style='color: #334155; font-size:0.75rem;'>DETECTOR.AI © 2026 • CREATED BY HIMANSHU</center>", unsafe_allow_html=True)