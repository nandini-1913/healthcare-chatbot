import streamlit as st
import json
import pickle
import numpy as np
import os
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedBot – Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── NLTK Setup ───────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk_resource(resource_paths, download_name):
    if isinstance(resource_paths, str):
        resource_paths = [resource_paths]
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue
    nltk.download(download_name, download_dir=NLTK_DATA_DIR, quiet=True)

ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource(["corpora/wordnet", "corpora/wordnet.zip"], "wordnet")

# ─── Load Model & Data ────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model   = load_model(os.path.join(BASE_DIR, "chatbot_model.keras"))
    intents = json.load(open(os.path.join(BASE_DIR, "intents_medquad.json")))
    words   = pickle.load(open(os.path.join(BASE_DIR, "words.pkl"), "rb"))
    classes = pickle.load(open(os.path.join(BASE_DIR, "classes.pkl"), "rb"))
    return model, intents, words, classes

model, intents, words, classes = load_resources()

# ─── NLP Helpers ──────────────────────────────────────────────────────────────
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_response(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return "I'm not sure I understood that. Could you describe your symptoms in more detail?"
    tag = classes[results[0][0]]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "I'm not sure I understood that. Please try again."

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2942 0%, #1a4a7a 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] .sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #ffffff !important;
    letter-spacing: -0.5px;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] .stat-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
}
[data-testid="stSidebar"] .stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.6;
}
[data-testid="stSidebar"] .stat-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #7dd3fc !important;
}

/* ── Main header ── */
.main-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #0f2942 0%, #1e5fa8 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(15,41,66,0.18);
}
.main-header .icon {
    font-size: 2.6rem;
    background: rgba(255,255,255,0.12);
    width: 60px; height: 60px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem !important;
    color: #ffffff !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 1.1;
}
.main-header p {
    color: #93c5fd !important;
    font-size: 0.85rem;
    margin: 3px 0 0 0;
}
.online-dot {
    width: 9px; height: 9px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Chat container ── */
.chat-container {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    height: 460px;
    overflow-y: auto;
    margin-bottom: 16px;
    scroll-behavior: smooth;
}
.chat-container::-webkit-scrollbar { width: 5px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    margin-bottom: 18px;
    align-items: flex-end;
    gap: 10px;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, #1e5fa8, #0f2942); color: white; }
.avatar.user { background: linear-gradient(135deg, #0ea5e9, #0284c7); color: white; }

.bubble {
    max-width: 72%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.55;
    position: relative;
}
.bubble.bot {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #e2e8f0;
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.bubble.user {
    background: linear-gradient(135deg, #1e5fa8, #0f2942);
    color: #ffffff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 12px rgba(30,95,168,0.25);
}
.timestamp {
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 4px;
    text-align: right;
}
.msg-row.user .timestamp { text-align: left; }

/* ── Welcome card ── */
.welcome-card {
    text-align: center;
    padding: 36px 20px;
    color: #64748b;
}
.welcome-card .big-icon { font-size: 3.5rem; margin-bottom: 12px; }
.welcome-card h3 {
    font-family: 'DM Serif Display', serif;
    color: #1e293b;
    font-size: 1.3rem;
    margin-bottom: 8px;
}
.welcome-card p { font-size: 0.88rem; line-height: 1.6; }

/* ── Quick suggestion chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.chip {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.chip:hover { background: #1e5fa8; color: white; border-color: #1e5fa8; }

/* ── Input area ── */
.stTextInput > div > div > input {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 0.92rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s !important;
    background: #ffffff !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1e5fa8 !important;
    box-shadow: 0 0 0 3px rgba(30,95,168,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e5fa8, #0f2942) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Disclaimer banner ── */
.disclaimer {
    background: #fef9c3;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #92400e;
    margin-top: 10px;
    display: flex;
    gap: 8px;
    align-items: flex-start;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🩺 MedBot</div>', unsafe_allow_html=True)
    st.markdown("*Your AI-powered healthcare assistant*")
    st.markdown("---")

    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Messages</div>
            <div class="stat-value">{st.session_state.msg_count}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Intents</div>
            <div class="stat-value">{len(classes)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
    - 💬 Type your symptoms or health question  
    - 🔍 Be specific for better answers  
    - 🔄 Ask follow-up questions freely  
    """)
    st.markdown("---")
    st.markdown("**Topics I can help with**")
    topics = ["🤒 Fever & Cold", "💊 Medications", "❤️ Heart Health",
              "🧠 Mental Health", "🦴 Joint Pain", "🩸 Diabetes", "🫁 Breathing"]
    for t in topics:
        st.markdown(f"<small>{t}</small>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.msg_count = 0
        st.rerun()

# ─── Main Area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="icon">🩺</div>
    <div>
        <h1>Healthcare Assistant</h1>
        <p><span class="online-dot"></span>Online · Powered by Deep Learning & NLP</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick suggestion chips ────────────────────────────────────────────────────
suggestions = ["I have a fever", "Chest pain", "Headache and nausea",
               "What is diabetes?", "I feel anxious", "Back pain relief"]

st.markdown('<div class="chip-row">' +
    "".join(f'<span class="chip">{s}</span>' for s in suggestions) +
    '</div>', unsafe_allow_html=True)

# ── Chat window ───────────────────────────────────────────────────────────────
chat_html = '<div class="chat-container" id="chat-box">'

if not st.session_state.messages:
    chat_html += """
    <div class="welcome-card">
        <div class="big-icon">👋</div>
        <h3>Hello! I'm MedBot</h3>
        <p>I can help you understand symptoms, medications, and general health questions.<br>
        Type a message below or click a suggestion to get started.</p>
    </div>"""
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        text = msg["text"]
        time = msg["time"]
        avatar = "🤖" if role == "bot" else "👤"
        chat_html += f"""
        <div class="msg-row {role}">
            <div class="avatar {role}">{avatar}</div>
            <div>
                <div class="bubble {role}">{text}</div>
                <div class="timestamp">{time}</div>
            </div>
        </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Auto-scroll to bottom
st.markdown("""
<script>
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
</script>
""", unsafe_allow_html=True)

# ── Input Row ─────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="message",
        placeholder="Describe your symptoms or ask a health question...",
        label_visibility="collapsed",
        key="user_input"
    )
with col_btn:
    send = st.button("Send ➤")

# ── Process message ───────────────────────────────────────────────────────────
if send and user_input.strip():
    now = datetime.now().strftime("%I:%M %p")

    st.session_state.messages.append({
        "role": "user",
        "text": user_input.strip(),
        "time": now
    })

    response = predict_response(user_input.strip())

    st.session_state.messages.append({
        "role": "bot",
        "text": response,
        "time": now
    })

    st.session_state.msg_count += 1
    st.rerun()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <span><strong>Medical Disclaimer:</strong> MedBot provides general health information only
    and is not a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical concerns.</span>
</div>
""", unsafe_allow_html=True)