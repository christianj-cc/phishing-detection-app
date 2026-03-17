# ============================================
# Cross-Platform Phishing Detection System
# Multi-Model Deployment (Defense Version)
# ============================================

import streamlit as st
import joblib
import numpy as np
import re
import string
import nltk
import time
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================
# NLTK SETUP
# ============================================

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ============================================
# STRUCTURAL FEATURES
# ============================================

def extract_email_features(text):
    return np.array([[
        len(re.findall(r'http\S+|www\S+', str(text))),
        str(text).count('!'),
        sum(1 for c in str(text) if c.isupper()),
        len(re.findall(r'\d', str(text))),
        len(str(text))
    ]])

def extract_sms_features(text):

    has_url = 1 if re.search(r'http\S+|www\S+', text) else 0
    has_email = 1 if re.search(r'\S+@\S+', text) else 0
    has_phone = 1 if re.search(r'\d{10,}', text) else 0

    text_length = len(text)

    upper_ratio = sum(1 for c in text if c.isupper()) / (text_length if text_length > 0 else 1)

    return np.array([[has_url, has_email, has_phone, text_length, upper_ratio]])

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_models():

    models = {}

    # EMAIL
    models["rf_email"] = joblib.load("rf_email.pkl")
    models["svm_email"] = joblib.load("svm_email.pkl")
    models["mlp_email"] = joblib.load("mlp_email.pkl")
    models["xgb_email"] = joblib.load("xgb_email.pkl")  
    models["hybrid_email"] = joblib.load("hybrid_email_opt.pkl")
    models["tfidf_email"] = joblib.load("tfidf_email.pkl")
    models["threshold_email"] = joblib.load("threshold_email.pkl")

    # SMS
    models["rf_sms"] = joblib.load("rf_sms.pkl")
    models["svm_sms"] = joblib.load("svm_sms.pkl")
    models["nn_sms"] = joblib.load("nn_sms.pkl")
    models["xgb_sms"] = joblib.load("xgb_sms_tuned.pkl")
    models["hybrid_sms"] = joblib.load("hybrid_sms_opt.pkl")
    models["tfidf_sms"] = joblib.load("tfidf_sms.pkl")
    models["threshold_sms"] = joblib.load("threshold_sms.pkl")

    return models
models = load_models()

# ============================================
# CUSTOM STYLING - Enhanced Dark UI
# ============================================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2f 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #a0a0c0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(30, 30, 46, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header i {
        color: #667eea;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(20, 20, 35, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
        padding: 0.8rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(25, 25, 45, 0.95) !important;
    }
    
    /* Label styling */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {
        color: #a0a0c0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(20, 20, 35, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        cursor: pointer !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Result box styling */
    .result-box {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-safe {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.15), rgba(0, 200, 100, 0.15));
        border: 1px solid #00ff87;
        color: #00ff87;
    }
    
    .result-phishing {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.15), rgba(200, 0, 0, 0.15));
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(20, 20, 35, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #667eea;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #a0a0c0;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px !important;
        height: 10px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2f 0%, #0f0f1e 100%) !important;
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(20, 20, 35, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 0.3em;
    }
    
    .stat-card:hover {
        border-color: #667eea;
        transform: scale(1.05);
    }
    
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #667eea;
    }
    
    .stat-label {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Divider styling */
    .custom-divider {
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        height: 2px;
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #66668a;
        font-size: 0.9rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 2rem;
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Indicator item styling */
    .indicator-item {
        background: rgba(20, 20, 35, 0.95);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .indicator-warning {
        border-left-color: #ff4b4b;
    }
    
    .indicator-safe {
        border-left-color: #00ff87;
    }
    
    /* Loading animation */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .loading-spinner i {
        color: #667eea;
        font-size: 2rem;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HARDCODED EVALUATION METRICS (from Colab)
# ============================================

METRICS = {
    "Email": {
        "Random Forest": {
            "accuracy": 98.64,
            "precision": 98.68,
            "recall": 98.71,
            "f1": 98.69,
            "roc_auc": 99.87,
            "fpr": 1.43,
            "fnr": 1.29
        },
        "SVM + XGB Ensemble": {
            "accuracy": 98.30,
            "precision": 98.11,
            "recall": 98.64,
            "f1": 98.37,
            "roc_auc": 99.84,
            "fpr": 2.06,
            "fnr": 1.36
        },
        "Neural Network": {
            "accuracy": 98.20,
            "precision": 98.63,
            "recall": 97.90,
            "f1": 98.26,
            "roc_auc": 99.75,
            "fpr": 1.48,
            "fnr": 2.10
        },
        "XGBoost": {
            "accuracy": 97.88,
            "precision": 97.31,
            "recall": 98.65,
            "f1": 97.97,
            "roc_auc": 99.78,
            "fpr": 2.95,
            "fnr": 1.35
        },
        "SVM": {
            "accuracy": 95.74,
            "precision": 99.71,
            "recall": 92.07,
            "f1": 95.74,
            "roc_auc": 99.78,
            "fpr": 0.29,
            "fnr": 7.93
        }
    },
    "SMS": {
        "SVM + XGB Ensemble": {
            "accuracy": 98.41,
            "precision": 96.83,
            "recall": 94.69,
            "f1": 95.75,
            "roc_auc": 99.66,
            "fpr": 0.72,
            "fnr": 5.31
        },
        "Neural Network": {
            "accuracy": 98.08,
            "precision": 95.93,
            "recall": 93.81,
            "f1": 94.85,
            "roc_auc": 99.43,
            "fpr": 0.93,
            "fnr": 6.19
        },
        "SVM": {
            "accuracy": 97.74,
            "precision": 99.50,
            "recall": 88.50,
            "f1": 93.68,
            "roc_auc": 99.49,
            "fpr": 0.10,
            "fnr": 11.50
        },
        "XGBoost": {
            "accuracy": 97.66,
            "precision": 96.26,
            "recall": 91.15,
            "f1": 93.64,
            "roc_auc": 99.37,
            "fpr": 0.83,
            "fnr": 8.85
        },
        "Random Forest": {
            "accuracy": 97.49,
            "precision": 95.79,
            "recall": 90.71,
            "f1": 93.18,
            "roc_auc": 99.35,
            "fpr": 0.93,
            "fnr": 9.29
        }
    }
}

# ============================================
# HEADER SECTION - Enhanced
# ============================================

st.markdown("""
<div style="text-align: center; padding: 1rem 0 1rem 0;">
    <div class="main-title">Cross-Platform Phishing Detection System</div>
    <div class="subtitle">Advanced machine learning protection for Email and SMS communications</div>
</div>
""", unsafe_allow_html=True)


# ============================================
# SIDEBAR CONFIG - With if/else for accuracy
# ============================================

with st.sidebar:
    st.markdown('<div class="sidebar-header">System Configuration</div>', unsafe_allow_html=True)
    
    platform = st.selectbox(
        "Platform",
        ["Email", "SMS"],
        help="Select the communication platform"
    )
    
    # Set default model based on platform
    model_options = ["SVM + XGB Ensemble", "Random Forest", "SVM", "Neural Network", "XGBoost"]
    if platform == "Email":
        default_model_index = 1  # Random Forest
    else:
        default_model_index = 0  # SVM + XGB Ensemble
    
    model_choice = st.selectbox(
        "Detection Model",
        model_options,
        index=default_model_index,
        help="Choose the ML model for detection"
    )
    
        # Instructions in styled expander
    with st.expander("Quick Guide", expanded=False):
        st.markdown("""
        <div style="color: #a0a0c0; font-size: 0.9rem;">
        1. Select platform (Email/SMS)<br>
        2. Choose detection model<br>
        3. Enter message content<br>
        4. Click 'Analyze'<br>
        5. Review results
        </div>
        """, unsafe_allow_html=True)
    
    # Add visual separator
    st.markdown('<hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.1);">', unsafe_allow_html=True)
    
    # Model Performance with real metrics
    st.markdown('<div class="sidebar-header">Model Performance</div>', unsafe_allow_html=True)
    
    # Get metrics for current selection
    m = METRICS[platform][model_choice]
    
    # Color thresholds for Recall and F1 (tuned to your data)
    def get_color(value, platform, metric):
        if platform == "Email":
            if metric == "recall":
                return "#00ff87" if value >= 98 else "#ffd700" if value >= 95 else "#ff4b4b"
            else:  # f1
                return "#00ff87" if value >= 98 else "#ffd700" if value >= 96 else "#ff4b4b"
        else:  # SMS
            if metric == "recall":
                return "#00ff87" if value >= 94 else "#ffd700" if value >= 90 else "#ff4b4b"
            else:  # f1
                return "#00ff87" if value >= 95 else "#ffd700" if value >= 93 else "#ff4b4b"
    
    recall_color = get_color(m['recall'], platform, "recall")
    f1_color = get_color(m['f1'], platform, "f1")
    
    # Two stat cards: Recall (replaces Models) and F1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: {recall_color};">{m['recall']:.1f}%</div>
            <div class="stat-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: {f1_color};">{m['f1']:.1f}%</div>
            <div class="stat-label">F1-score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Smaller row for Precision and Accuracy
    st.markdown("""
    <style>
    .small-metric-box {
        background: rgba(20, 20, 35, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.4rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .small-metric-box .value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1.2;
    }
    .small-metric-box .label {
        font-size: 0.7rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="small-metric-box">
            <div class="value">{m['precision']:.1f}%</div>
            <div class="label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="small-metric-box">
            <div class="value">{m['accuracy']:.1f}%</div>
            <div class="label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add help section
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
        - **F1-score**: Harmonic mean of precision and recall (0-100%). Higher is better.
        - **Precision**: When the model says "phishing", how often is it correct?
        - **Recall**: What percentage of actual phishing messages did the model catch?
        - **Accuracy**: Overall correct predictions (can be misleading with imbalanced data)
        """)
    
    st.markdown('<hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.1);">', unsafe_allow_html=True)
    
    # Analysis button
    analyze_button = st.button("Analyze Message", use_container_width=True)
    

# ============================================
# DETECT INDICATORS (Unchanged)
# ============================================

def detect_indicators(text):
    indicators = []
    
    if re.search(r'http\S+|www\S+', text):
        indicators.append("URL detected")
    
    if sum(1 for c in text if c.isupper()) > 20:
        indicators.append("Excessive uppercase")
    
    if "urgent" in text.lower():
        indicators.append("Urgency keyword")
    
    if "verify" in text.lower():
        indicators.append("Account verification request")
    
    if len(text) > 500:
        indicators.append("Unusually long message")
    
    return indicators


# ============================================
# MAIN DASHBOARD LAYOUT
# ============================================

# Create tabs for better organization
tab1, tab2 = st.tabs(["Message Analysis", "ⓘ About the System"])

with tab1:
    # Main analysis area
    col1, col2 = st.columns([1.2, 0.8])

    # ============================================
    # INPUT PANEL - Enhanced
    # ============================================

    with col1:
        st.markdown('<div class="section-header"><span>Message Input</span></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            if platform == "Email":
                sender_email = st.text_input("From", placeholder="sender@example.com")
                email_subject = st.text_input("Subject", placeholder="Enter email subject")
                email_body = st.text_area(
                    "Message Body",
                    height=160,
                    placeholder="Paste email content here..."
                )
                email_date = st.text_input("Date (optional)", placeholder="YYYY-MM-DD")
                user_input = f"{sender_email} {email_subject} {email_body} {email_date}"
            else:
                user_input = st.text_area(
                    "SMS Message",
                    height=220,
                    placeholder="Enter SMS message to analyze..."
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ============================================
    # RESULT PANEL - Enhanced
    # ============================================

    with col2:
        st.markdown('<div class="section-header"><span>Detection Result</span></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            result_box = st.empty()
            model_box = st.empty()
            confidence_box = st.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ====================================
            # EXPLAINABILITY OUTPUT - Enhanced
            # ====================================

            indicators = detect_indicators(user_input)

        with st.container():
            if indicators:
                for i in indicators:
                    st.markdown(f"""
                    <div class="indicator-item indicator-warning">
                        ⚠️ {i}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="indicator-item indicator-safe">
                    ✅ No strong phishing indicators detected
                </div>
                """, unsafe_allow_html=True)
            
            


    # ============================================
    # INDICATORS PANEL - Enhanced
    # ============================================

    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        indicator_box = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # About section
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("""
    ### About the System
    
    This advanced phishing detection system uses ensemble machine learning to protect against malicious emails and SMS messages.
    
    **Key Features:**
    - Multi-model ensemble approach
    - Real-time analysis
    - Detailed suspicious indicator detection
    - Support for both Email and SMS platforms
    - High accuracy with 5 different ML models
    
    **Models Used:**
    - Random Forest
    - Support Vector Machine (SVM)
    - Neural Network
    - XGBoost
    - SVM + XGBoost Ensemble
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION (Completely Unchanged)
# ============================================

if analyze_button:
    if not user_input.strip():
        result_box.warning("Please enter a message to analyze.")
    else:
        # Show loading animation
        with st.spinner("Analyzing message..."):
            time.sleep(1)  # Small delay for better UX
            
            cleaned = clean_text(user_input)

            if platform == "Email":
                tfidf = models["tfidf_email"]
                tfidf_features = tfidf.transform([cleaned])
                struct_features = extract_email_features(user_input)
                X_final = hstack([tfidf_features, struct_features])

                if model_choice == "Random Forest":
                    model = models["rf_email"]
                    threshold = 0.5
                elif model_choice == "SVM":
                    model = models["svm_email"]
                    threshold = 0.0
                elif model_choice == "Neural Network":
                    model = models["mlp_email"]
                    threshold = 0.5
                elif model_choice == "XGBoost":
                    model = models["xgb_email"]
                    threshold = 0.5
                else:
                    model = models["hybrid_email"]
                    threshold = models["threshold_email"]

            else:
                tfidf = models["tfidf_sms"]
                tfidf_features = tfidf.transform([cleaned])
                struct_features = extract_sms_features(user_input)
                X_final = hstack([tfidf_features, struct_features])

                if model_choice == "Random Forest":
                    model = models["rf_sms"]
                    threshold = 0.5
                elif model_choice == "SVM":
                    model = models["svm_sms"]
                    threshold = 0.0
                elif model_choice == "Neural Network":
                    model = models["nn_sms"]
                    threshold = 0.5
                elif model_choice == "XGBoost":
                    model = models["xgb_sms"]
                    threshold = 0.5
                else:
                    model = models["hybrid_sms"]
                    threshold = models["threshold_sms"]

            # ====================================
            # Prediction
            # ====================================

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_final)[0][1]
                prediction = 1 if prob >= threshold else 0
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X_final)[0]
                prediction = 1 if score >= threshold else 0
                prob = float(score)
            else:
                prediction = model.predict(X_final)[0]
                prob = None

            # ====================================
            # DISPLAY RESULTS - Enhanced
            # ====================================

            with result_box.container():
                if prediction == 1:
                    st.markdown("""
                    <div class="result-box result-phishing">
                        PHISHING MESSAGE DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box result-safe">
                        LEGITIMATE MESSAGE
                    </div>
                    """, unsafe_allow_html=True)

            with model_box.container():
                st.markdown(f"""
                <div class="info-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a0a0c0;">Platform:</span>
                        <span style="color: #667eea; font-weight: 600;">{platform}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
                        <span style="color: #a0a0c0;">Model:</span>
                        <span style="color: #667eea; font-weight: 600;">{model_choice}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with confidence_box.container():
                if prob is not None and model_choice != "SVM":
                    prob_display = float(min(max(prob,0),1))
                    
                    st.markdown("""
                    <div style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="color: #a0a0c0;">Phishing Probability</span>
                            <span style="color: #667eea; font-weight: 600;">{:.1f}%</span>
                        </div>
                    </div>
                    """.format(prob_display*100), unsafe_allow_html=True)
                    
                    st.progress(prob_display)

# ============================================
# FOOTER - Enhanced
# ============================================

st.markdown("""
<div class="footer">
    <div style="margin-bottom: 0.5rem;">🛡️ Cross-Platform Phishing Detection System | Thesis Deployment</div>
    <div style="font-size: 0.8rem; opacity: 0.7;">Version 2.0 | Powered by Ensemble Machine Learning</div>
</div>
""", unsafe_allow_html=True)
