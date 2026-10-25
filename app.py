import os
import re
import io
import sys
import json
import time
import uuid
import shutil
import hashlib
import html as html_lib
import tempfile
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv
from gtts import gTTS
from ddgs import DDGS


# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Galaxy AI Chat",
    page_icon="🌌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CHAT_HISTORY_FILE = BASE_DIR / "chat_history.json"
MEMORY_FILE = BASE_DIR / "ai_memory.json"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()

MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# Audio sathi changle models
TRANSCRIBE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

PDF_CHAR_LIMIT = 20000

LANGUAGES = [
    "Auto Detect",
    "Marathi",
    "Hindi",
    "English",
    "Bengali",
    "Gujarati",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Punjabi",
    "Urdu",
    "French",
    "Spanish",
    "German",
    "Japanese",
    "Korean",
    "Chinese",
    "Arabic",
    "Portuguese",
    "Russian",
    "Italian",
]

TTS_LANGUAGES = {
    "Marathi": "mr",
    "Hindi": "hi",
    "English": "en",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
}

VOICE_INPUT_LANGUAGES = [
    "Marathi",
    "Hindi",
    "English",
    "Auto Detect",
]


# ==========================================
# EXTRA CONFIGURATION
# ==========================================

CHATS_FILE = BASE_DIR / "chats.json"
LEGACY_CHAT_FILE = CHAT_HISTORY_FILE  # junya version cha chat_history.json
SETTINGS_FILE = BASE_DIR / "settings.json"

VIEWS = [
    "💬 Chat",
    "💻 Code Workspace",
    "🔎 Search Chats",
    "🔒 Privacy & Data",
]

AI_MODES = [
    "General",
    "Python Coding",
    "C Programming",
    "Cyber Security",
    "AI Automation",
    "Code Error Fixer",
    "Code Explainer",
]

MODE_RULES = {
    "Code Error Fixer": (
        "Find errors in the user's Python or C code. For every error give "
        "the line number, the cause in simple words, and the fix. Then give "
        "the complete corrected code in one code block."
    ),
    "Code Explainer": (
        "Explain the user's code line by line in simple language, then "
        "summarize what the whole program does."
    ),
}

DEFAULT_SETTINGS = {
    "save_history": True,
    "use_memory": True,
    "auto_delete_days": 0,
}

PDF_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\mangal.ttf",
    r"C:\Windows\Fonts\Nirmala.ttf",
    r"C:\Windows\Fonts\arialuni.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


# ==========================================
# WEB SEARCH
# ==========================================

def web_search(query, max_results=5):
    """Returns a list of {title, body, url} dicts (empty list on failure)."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []

    results = []
    for item in raw:
        url = item.get("href", "") or ""
        if not url.startswith(("http://", "https://")):
            continue
        results.append({
            "title": item.get("title", "No title") or "No title",
            "body": item.get("body", "") or "",
            "url": url,
        })
    return results


def format_search_results(results):
    if not results:
        return "No search results found."

    return "\n\n".join(
        f"[{i}] Title: {r['title']}\nSummary: {r['body']}\nSource: {r['url']}"
        for i, r in enumerate(results, start=1)
    )


# ==========================================
# JSON / FILE HELPERS
# ==========================================

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, data):
    try:
        temp_file = Path(path).with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, path)
        return True
    except OSError:
        return False


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# ==========================================
# CHAT DATA MODEL
# ==========================================
# chats = {chat_id: {id, title, created, updated, incognito, messages}}
# message = {role, content, feedback?, sources?}

def new_chat(incognito=False):
    return {
        "id": uuid.uuid4().hex[:10],
        "title": "New chat",
        "created": now_iso(),
        "updated": now_iso(),
        "incognito": incognito,
        "messages": [],
    }


def clean_messages(raw):
    cleaned = []
    for m in raw if isinstance(raw, list) else []:
        if not (
            isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ):
            continue

        item = {"role": m["role"], "content": m["content"]}

        if m.get("feedback") in ("up", "down"):
            item["feedback"] = m["feedback"]

        if isinstance(m.get("sources"), list):
            item["sources"] = [
                s for s in m["sources"]
                if isinstance(s, dict) and s.get("url")
            ]

        cleaned.append(item)
    return cleaned


def load_chats():
    chats = {}
    data = load_json(CHATS_FILE, None)

    if isinstance(data, dict) and isinstance(data.get("chats"), dict):
        for cid, c in data["chats"].items():
            if not isinstance(c, dict):
                continue
            chat = new_chat()
            chat["id"] = str(cid)
            for field in ("title", "created", "updated"):
                if isinstance(c.get(field), str):
                    chat[field] = c[field]
            chat["messages"] = clean_messages(c.get("messages"))
            chats[chat["id"]] = chat

    elif LEGACY_CHAT_FILE.exists():
        # Junya chat_history.json ek navin chat mhanun import kara
        old = clean_messages(load_json(LEGACY_CHAT_FILE, []))
        if old:
            chat = new_chat()
            chat["title"] = "Old chat"
            chat["messages"] = old
            chats[chat["id"]] = chat

    return chats


def save_chats():
    if not st.session_state.settings.get("save_history", True):
        return False

    data = {
        "chats": {
            cid: c
            for cid, c in st.session_state.chats.items()
            if not c.get("incognito")
        }
    }
    return write_json(CHATS_FILE, data)


def save_settings():
    return write_json(SETTINGS_FILE, st.session_state.settings)


def purge_old_chats(days, keep_id=None):
    """Delete chats not updated for `days` days. Returns count removed."""
    if days <= 0:
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    removed = 0

    for cid in list(st.session_state.chats):
        if cid == keep_id:
            continue
        try:
            updated = datetime.fromisoformat(
                st.session_state.chats[cid]["updated"]
            )
        except (ValueError, KeyError):
            continue
        if updated < cutoff:
            del st.session_state.chats[cid]
            removed += 1

    return removed


def ensure_current_chat():
    chats = st.session_state.chats

    if not chats:
        chat = new_chat()
        chats[chat["id"]] = chat

    if st.session_state.get("current_chat_id") not in chats:
        st.session_state.current_chat_id = max(
            chats.values(), key=lambda c: c["updated"]
        )["id"]


def current_chat():
    ensure_current_chat()
    return st.session_state.chats[st.session_state.current_chat_id]


# ==========================================
# SESSION STATE
# ==========================================

DEFAULT_STATE = {
    "working_model": None,
    "last_audio_hash": "",
    "voice_draft": "",
    "voice_draft_active": False,
    "pending_voice_prompt": "",
    "last_answer": "",
    "last_audio_output": None,
    "galaxy_pdf_text": "",
    "galaxy_pdf_pages_list": [],
    "galaxy_pdf_name": "",
    "galaxy_pdf_pages": 0,
    "pdf_uploader_key": 0,
    "img_uploader_key": 0,
    "generated_image": None,
    "verify_result": None,
    "view": VIEWS[0],
    "ws_code": "print('Hello, Galaxy AI!')\n",
    "ws_output": "",
    "ws_ai_answer": "",
    "ws_fixed_code": "",
    "ws_project_answer": "",
}

for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

if "settings" not in st.session_state:
    _saved = load_json(SETTINGS_FILE, {})
    st.session_state.settings = {
        **DEFAULT_SETTINGS,
        **(_saved if isinstance(_saved, dict) else {}),
    }

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
    purge_old_chats(st.session_state.settings.get("auto_delete_days", 0))
    ensure_current_chat()
    save_chats()

if "ai_memories" not in st.session_state:
    _mem = load_json(MEMORY_FILE, [])
    st.session_state.ai_memories = _mem if isinstance(_mem, list) else []


def save_memories():
    return write_json(MEMORY_FILE, st.session_state.ai_memories)


# ==========================================
# GEMINI CLIENT
# ==========================================

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# ==========================================
# GEMINI API + AUTOMATIC FALLBACK
# ==========================================

def gemini_generate(contents, fast=True, model_order=None, temperature=None):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing.")

    client = get_gemini_client(API_KEY)

    if model_order:
        models = list(model_order)
    else:
        previous_model = st.session_state.get("working_model")
        models = (
            ([previous_model] if previous_model else [])
            + [m for m in MODEL_CANDIDATES if m != previous_model]
        )

    last_error = None
    tried = []

    for model_name in models:
        config_args = {}

        # Temperature fakt junya 2.5 models la.
        if temperature is not None and model_name.startswith("gemini-2.5"):
            config_args["temperature"] = temperature

        # thinking_budget fakt 2.5 models sathi aahe
        if fast and model_name.startswith("gemini-2.5"):
            try:
                config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            except Exception:
                pass

        config = (
            types.GenerateContentConfig(**config_args)
            if config_args
            else None
        )

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )

                if not model_order:
                    st.session_state.working_model = model_name
                return response

            except errors.APIError as error:
                last_error = error
                tried.append(f"{model_name}: {error.code}")

                # Temporary error -> ekda parat try kara
                if error.code in (429, 500, 502, 503, 504):
                    if attempt == 0:
                        time.sleep(1)
                        continue
                    break

                # Bad request -> config nakka karun ekda parat try kara
                if error.code == 400 and config is not None and attempt == 0:
                    config = None
                    continue

                # 404 (model nahi), 401/403 -> pudhcha model
                break

    if not model_order:
        st.session_state.working_model = None

    if last_error:
        raise RuntimeError(
            "Konatahi model chalala nahi. Tried -> "
            + ", ".join(tried)
            + f" | Last error: {last_error}"
        )

    raise RuntimeError("Kontahi Gemini model available nahi.")


# ==========================================
# AUDIO TRANSCRIPTION
# ==========================================

def transcribe_audio(audio_bytes, spoken_language, mime_type="audio/wav"):
    language_hints = {
        "Auto Detect": (
            "The speaker most likely speaks Marathi, Hindi, or English, "
            "often mixed. Identify the language carefully."
        ),
        "Marathi": (
            "The speaker is speaking Marathi (may include English "
            "words, e.g. programming terms). Write Marathi in "
            "Devanagari script."
        ),
        "Hindi": (
            "The speaker is speaking Hindi (may include English words). "
            "Write Hindi in Devanagari script."
        ),
        "English": "The speaker is speaking English.",
    }

    instruction = f"""
You are a precise speech-to-text engine.

Transcribe ONLY the main, closest, loudest speaker in the audio.
Ignore background noise, music, TV, other people's voices,
keyboard sounds, and silence.
Do not guess or invent words. If a part is unclear, skip it.
Do not answer the spoken question.
Do not translate it.
Do not add explanations.

{language_hints.get(spoken_language, language_hints["Auto Detect"])}

Return only the transcription.
If there is no clear speech, return an empty response.
"""

    response = gemini_generate(
        [
            instruction,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        model_order=TRANSCRIBE_MODELS,
        temperature=0,
    )

    return (response.text or "").strip()


# ==========================================
# TEXT-TO-SPEECH
# ==========================================

def clean_for_speech(text):
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*#`_>]", "", text)
    return text.strip()


def get_tts_language(text, selected_language):
    if selected_language != "Auto Detect":
        return TTS_LANGUAGES.get(selected_language, "en")

    # Basic script detection. This is approximate.
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "pa"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gu"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "te"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kn"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "ml"
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"
    if re.search(r"[\u0900-\u097F]", text):
        # Devanagari may be Marathi, Hindi, or another language.
        # Marathi-specific detection is not possible here, so use Hindi.
        return "hi"

    return "en"


def make_speech(text, selected_language):
    clean_text = clean_for_speech(text)

    if not clean_text:
        return None

    language_code = get_tts_language(clean_text, selected_language)

    audio_buffer = io.BytesIO()
    tts = gTTS(text=clean_text[:4500], lang=language_code)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# ==========================================
# VOICE DRAFT CALLBACK
# ==========================================

def send_voice_draft():
    text = st.session_state.get("voice_draft", "").strip()

    if text:
        st.session_state.pending_voice_prompt = text

    st.session_state.voice_draft = ""
    st.session_state.voice_draft_active = False


# ==========================================
# GALAXY UI
# ==========================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 20%,
                rgba(87,40,170,0.22),
                transparent 40%
            ),
            radial-gradient(
                ellipse at 85% 75%,
                rgba(0,125,190,0.16),
                transparent 40%
            ),
            linear-gradient(135deg, #03040c, #080b1c 55%, #03040c);
        color: #f4f4ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 14rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d20, #10132a);
        border-right: 1px solid rgba(160,140,255,0.20);
    }

    [data-testid="stSidebar"] * {
        color: #eeeeff;
    }

    h1, h2, h3 {
        color: #f5f1ff !important;
    }

    h1 {
        font-weight: 800 !important;
        text-shadow: 0 0 24px rgba(153,110,255,0.35);
    }

    [data-testid="stChatMessage"] {
        background: rgba(12,16,35,0.96) !important;
        border: 1px solid #514580 !important;
        border-radius: 18px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #f5f5ff !important;
    }

    .stTextInput input {
        background: #10142a !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
        border: 1px solid #8065e8 !important;
        border-radius: 12px !important;
    }

    /* Voice bar: small, fixed ABOVE the typing box */
    .st-key-voice_bar {
        position: fixed;
        bottom: 8rem;
        left: 0;
        right: 0;
        margin-left: auto;
        margin-right: auto;
        width: min(760px, 92vw);
        max-height: 28vh;
        overflow-y: auto;
        z-index: 999;
        padding: 6px 10px 4px 10px;
        background: rgba(12,16,35,0.97);
        border: 1px solid #514580;
        border-radius: 14px;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.4);
    }

    /* Sidebar open asel tar bar main area chya madhyat rahil */
    [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"]
    .st-key-voice_bar {
        left: 21rem;
    }

    .st-key-voice_bar [data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }

    .st-key-voice_bar [data-testid="stHorizontalBlock"] {
        gap: 0.5rem;
        align-items: center;
    }

    .st-key-voice_bar label,
    .st-key-voice_bar p {
        color: #eeeeff !important;
        font-size: 0.8rem;
    }

    .st-key-voice_bar [data-testid="stSelectbox"] > div > div {
        min-height: 2.2rem;
    }

    .st-key-voice_bar [data-testid="stAudioInput"] {
        transform: scale(0.88);
        transform-origin: left center;
    }

    /* Bottom bar (pinned chat box) */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        background: #10142a !important;
        border: 1px solid #8065e8 !important;
        border-radius: 14px !important;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"] {
        background: #10142a !important;
        border-color: #8065e8 !important;
    }

    [data-testid="stAudioInput"],
    [data-testid="stAudioInput"] > div {
        background: #151a35 !important;
        border-radius: 12px !important;
        color: #eeeeff !important;
    }

    [data-testid="stChatInput"] textarea {
        color: white !important;
        -webkit-text-fill-color: white !important;
        max-height: 220px;
    }

    .stTextArea textarea {
        background: #10142a !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
        border: 1px solid #8065e8 !important;
        border-radius: 12px !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7048df, #4169e1);
        color: white !important;
        border: 1px solid rgba(190,170,255,0.35);
        border-radius: 12px;
        font-weight: 600;
    }

    [data-testid="stSelectbox"] > div > div {
        background: #151a35 !important;
        color: white !important;
    }

    /* ===== COLOURFUL SIDEBAR OPTIONS ===== */

    /* View options (Chat / Code Workspace / Search / Privacy) */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background: linear-gradient(135deg, rgba(112,72,223,0.35), rgba(65,105,225,0.35));
        border: 1px solid rgba(190,170,255,0.35);
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 6px;
        width: 100%;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"]:nth-of-type(1) {
        background: linear-gradient(135deg, #7048df, #4169e1);
    }
    [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #00c6ff, #7a5cff) !important;
        border: 1px solid #ffffff;
        box-shadow: 0 0 12px rgba(0,198,255,0.55);
        font-weight: 700;
    }

    /* New chat / Clear / Delete */
    .st-key-new_chat_btn button {
        background: linear-gradient(135deg, #00c853, #00b0ff) !important;
        border: 1px solid #b9ffe0 !important;
        font-size: 1.05rem;
        box-shadow: 0 0 14px rgba(0,200,120,0.45);
    }
    .st-key-clear_chat_btn button {
        background: linear-gradient(135deg, #ff9800, #ff5722) !important;
    }
    .st-key-delete_chat_btn button {
        background: linear-gradient(135deg, #e53935, #8e0038) !important;
    }

    /* Chat list: normal chats = blue/purple, current chat = green */
    [class*="st-key-chat_pick_"] button {
        background: linear-gradient(135deg, #3a47d5, #7b2ff7) !important;
        justify-content: flex-start;
    }
    [class*="st-key-chat_current_"] button {
        background: linear-gradient(135deg, #00b09b, #96c93d) !important;
        border: 1px solid #ffffff !important;
        box-shadow: 0 0 12px rgba(0,230,150,0.5);
        justify-content: flex-start;
    }

    /* Expanders (PDF, Image, Learning, Export ...) */
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        background: linear-gradient(135deg, rgba(255,0,128,0.14), rgba(0,200,255,0.14));
        border: 1px solid rgba(255,140,220,0.45) !important;
        border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-weight: 700;
    }

    /* Feedback + other buttons keep the galaxy gradient */
    </style>

    /* Mobile fix: do not pin the voice bar on small screens */
    @media (max-width: 768px) {
        .st-key-voice_bar {
            position: static !important;
            width: 100% !important;
            max-height: none !important;
            margin-bottom: 1rem;
        }
    }
    """,
    unsafe_allow_html=True,
)


# ==========================================
# AI HELPERS
# ==========================================

def ask_ai(prompt):
    response = gemini_generate(prompt)
    return (response.text or "").strip()


def extract_code_block(text):
    match = re.search(r"```[\w+#.-]*\n(.*?)```", text, flags=re.S)
    return (match.group(1).rstrip() + "\n") if match else ""


def build_system_prompt(language, topic, question, learning_on, learning_level):
    if language == "Auto Detect":
        language_rule = """
Detect the user's language and reply in that language.
Understand Roman Marathi, Hindi, English, and common typing mistakes.
"""
    else:
        language_rule = f"""
Reply in {language}, unless the user explicitly requests another language.
Understand Roman Marathi and common typing mistakes.
"""

    system_prompt = f"""
You are Galaxy AI, a helpful AI assistant.

Selected AI mode: {topic}

LANGUAGE:
{language_rule}

Give clear and useful beginner-friendly answers.
For coding requests, provide working code and simple explanations.
In Cyber Security mode, focus on authorized and ethical security work.
In AI Automation mode, assist with Python, chatbots, and workflow automation.
{MODE_RULES.get(topic, "")}

You may help the user analyze documents they upload, including
extracting or summarizing information when appropriate.
Respect privacy, avoid unnecessary disclosure of sensitive information,
and never invent details that are absent from the document.
"""

    if learning_on:
        system_prompt += f"""
LEARNING MODE ({learning_level} level):
Teach the topic step by step.
- If this is a new topic, start with a short roadmap (3-6 steps), then teach
  only step 1 with a simple example.
- In later replies teach only the next step.
- End every reply with one small practice question or mini quiz.
- If the user answered your previous question, check it kindly, explain any
  mistake, and then move to the next step.
"""

    # Memories
    if (
        st.session_state.settings.get("use_memory", True)
        and st.session_state.ai_memories
    ):
        system_prompt += (
            "\nThings to remember about the user:\n"
            + "\n".join(f"- {m}" for m in st.session_state.ai_memories)
            + "\n"
        )

    # PDF content (relevant parts only when the PDF is large)
    pdf_context = retrieve_pdf_context(question)

    if pdf_context.strip():
        system_prompt += (
            "\nThe user uploaded a PDF."
            "\nAnswer questions using ONLY the actual extracted PDF text below."
            "\nMention the page number like (Page 3) when you use it."
            "\nIf the answer is not in the text, say so clearly."
            "\nDo not invent missing content or use placeholders."
            "\n\nPDF CONTENT:\n"
            + pdf_context
            + "\n"
        )

    return system_prompt


# ==========================================
# PDF QUESTION ANSWERING (retrieval)
# ==========================================

def build_pdf_chunks(pages, size=1500, overlap=200):
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        text = (text or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            chunks.append((page_no, text[start:start + size]))
            if start + size >= len(text):
                break
            start += size - overlap

    return chunks


def tokenize(text):
    words = re.split(r"[\s\.,;:!?()\[\]{}\"'“”‘’|/\\\-]+", text.lower())
    return {w for w in words if len(w) >= 3}


def retrieve_pdf_context(question, limit=PDF_CHAR_LIMIT):
    pages = st.session_state.get("galaxy_pdf_pages_list") or []

    full = "\n".join(
        f"[Page {i}]\n{t.strip()}"
        for i, t in enumerate(pages, start=1)
        if t and t.strip()
    )

    if len(full) <= limit:
        return full

    chunks = build_pdf_chunks(pages)
    q_words = tokenize(question or "")

    if not q_words:
        return full[:limit]

    scored = sorted(
        ((len(q_words & tokenize(text)), i) for i, (_, text) in enumerate(chunks)),
        reverse=True,
    )

    picked = []
    total = 0

    for score, i in scored:
        if score == 0 and picked:
            break
        page_no, text = chunks[i]
        if total + len(text) > limit:
            continue
        picked.append((i, page_no, text))
        total += len(text)

    if not picked:
        return full[:limit]

    picked.sort()  # document order
    return "\n\n".join(f"[Page {p}]\n{t}" for _, p, t in picked)


# ==========================================
# SOURCE VERIFICATION
# ==========================================

def verify_answer(question, answer):
    query = (question or answer).strip().replace("\n", " ")[:150]
    results = web_search(query, max_results=6)

    if not results:
        return (
            "❓ Internet search madhye kahi milale nahi, "
            "mhanun answer verify karta aale nahi.",
            [],
        )

    prompt = f"""
You are a careful fact-checker.
Check the ANSWER below against the SEARCH RESULTS only.
Do not use outside knowledge.

For every important factual claim write one line:
- ✅ Supported — short reason [source number]
- ⚠️ Partly supported — what is missing or different [source number]
- ❌ Contradicted — what the source says [source number]
- ❓ Cannot verify — not found in the results

Finish with:
**Overall reliability:** High / Medium / Low, and one sentence why.

Reply in the same language as the ANSWER.

QUESTION:
{question}

ANSWER TO CHECK:
{answer}

SEARCH RESULTS:
{format_search_results(results)}
"""

    return ask_ai(prompt), results


# ==========================================
# CHAT SEARCH
# ==========================================

def search_chats(query, limit=60):
    q = query.lower().strip()
    hits = []

    if not q:
        return hits

    chats = sorted(
        st.session_state.chats.values(),
        key=lambda c: c["updated"],
        reverse=True,
    )

    for chat in chats:
        if q in chat["title"].lower():
            hits.append((chat, -1, "title", chat["title"]))

        for idx, message in enumerate(chat["messages"]):
            content = message["content"]
            pos = content.lower().find(q)

            if pos >= 0:
                start = max(0, pos - 60)
                snippet = content[start:pos + len(q) + 100].replace("\n", " ")
                hits.append((chat, idx, message["role"], snippet))

            if len(hits) >= limit:
                return hits

    return hits


# ==========================================
# CHAT EXPORT
# ==========================================

def chat_to_text(chat):
    lines = [chat["title"], f"Created: {chat['created']}", ""]

    for m in chat["messages"]:
        lines.append(f"{m['role'].upper()}: {m['content']}")
        for s in m.get("sources", []) or []:
            lines.append(f"  Source: {s.get('title', '')} - {s.get('url', '')}")
        lines.append("")

    return "\n".join(lines)


def chat_to_markdown(chat):
    lines = [f"# {chat['title']}", f"*Created: {chat['created']}*", ""]

    for m in chat["messages"]:
        who = "🧑 You" if m["role"] == "user" else "🌌 Galaxy AI"
        lines.append(f"### {who}")
        lines.append(m["content"])
        for s in m.get("sources", []) or []:
            lines.append(f"- [{s.get('title', 'Source')}]({s.get('url', '')})")
        lines.append("")

    return "\n".join(lines)


def chat_to_html(chat):
    esc = html_lib.escape
    blocks = []

    for m in chat["messages"]:
        who = "You" if m["role"] == "user" else "Galaxy AI"
        sources = "".join(
            f'<li><a href="{esc(s.get("url", ""))}">{esc(s.get("title", "Source"))}</a></li>'
            for s in (m.get("sources") or [])
        )
        sources_html = f"<ul>{sources}</ul>" if sources else ""
        blocks.append(
            f'<div class="msg {m["role"]}"><b>{who}</b>'
            f'<pre>{esc(m["content"])}</pre>{sources_html}</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="mr"><head><meta charset="utf-8">
<title>{esc(chat["title"])}</title>
<style>
body {{ font-family: system-ui, "Noto Sans Devanagari", sans-serif;
        max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
.msg {{ border: 1px solid #999; border-radius: 12px; padding: 12px;
        margin: 12px 0; }}
.msg.user {{ background: #f1edff; }}
pre {{ white-space: pre-wrap; font-family: inherit; margin: 6px 0 0; }}
</style></head><body>
<h1>{esc(chat["title"])}</h1>
<p><small>Created: {esc(chat["created"])}</small></p>
{"".join(blocks)}
</body></html>"""


def chat_to_pdf(chat):
    """Returns PDF bytes, or None if reportlab is missing / build fails."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return None

    font_name = "Helvetica"
    for path in PDF_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("GalaxyFont", path))
                font_name = "GalaxyFont"
                break
            except Exception:
                continue

    def prep(text):
        if font_name == "Helvetica":
            text = text.encode("latin-1", "replace").decode("latin-1")
        return xml_escape(text).replace("\n", "<br/>")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=styles["BodyText"],
        fontName=font_name, fontSize=10, leading=14,
    )
    label = ParagraphStyle(
        "label", parent=body, fontSize=11, spaceBefore=8, spaceAfter=2,
    )
    title = ParagraphStyle(
        "title", parent=styles["Title"], fontName=font_name,
    )

    story = [Paragraph(prep(chat["title"]), title), Spacer(1, 10)]

    for m in chat["messages"]:
        who = "YOU" if m["role"] == "user" else "GALAXY AI"
        story.append(Paragraph(prep(who), label))
        story.append(Paragraph(prep(m["content"]), body))
        for s in m.get("sources", []) or []:
            story.append(Paragraph(
                prep(f"Source: {s.get('title', '')} - {s.get('url', '')}"),
                body,
            ))
        story.append(Spacer(1, 6))

    try:
        buffer = io.BytesIO()
        SimpleDocTemplate(buffer, pagesize=A4).build(story)
        return buffer.getvalue()
    except Exception:
        return None


# ==========================================
# CODE RUNNER (Python / C)
# ==========================================

def run_code(code, language, stdin_text="", timeout=10):
    """Runs code locally. Returns (success, output_text)."""
    max_output = 6000

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)

        try:
            if language == "Python":
                src = tmp / "main.py"
                src.write_text(code, encoding="utf-8")
                cmd = [sys.executable, str(src)]

            else:
                compiler = shutil.which("gcc") or shutil.which("cc")
                if not compiler:
                    return False, (
                        "gcc compiler milala nahi. C code chalavnyasathi "
                        "gcc install kara (Windows: MinGW / MSYS2)."
                    )

                src = tmp / "main.c"
                src.write_text(code, encoding="utf-8")
                exe = tmp / ("main.exe" if os.name == "nt" else "main")

                compiled = subprocess.run(
                    [compiler, str(src), "-o", str(exe)],
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=30,
                )
                if compiled.returncode != 0:
                    return False, (
                        "COMPILE ERROR:\n"
                        + (compiled.stderr or compiled.stdout)[:max_output]
                    )
                cmd = [str(exe)]

            result = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=tmp,
            )

            output = result.stdout or ""
            if result.stderr:
                output += ("\n" if output else "") + "STDERR:\n" + result.stderr
            if not output:
                output = "(no output)"
            output += f"\n\n[exit code: {result.returncode}]"

            return result.returncode == 0, output[:max_output]

        except subprocess.TimeoutExpired:
            return False, (
                f"Timeout: code {timeout} seconds peksha jast velach chalat hota "
                "(infinite loop ki input pahije?)."
            )
        except Exception as error:
            return False, f"Run error: {error}"


# ==========================================
# CALLBACKS
# ==========================================

def new_chat_cb():
    chat = new_chat()
    st.session_state.chats[chat["id"]] = chat
    st.session_state.current_chat_id = chat["id"]
    st.session_state.verify_result = None
    st.session_state.last_answer = ""
    save_chats()


def clear_chat_cb():
    chat = current_chat()
    chat["messages"] = []
    chat["title"] = "New chat"
    chat["updated"] = now_iso()
    st.session_state.verify_result = None
    st.session_state.last_answer = ""
    st.session_state.last_audio_output = None
    save_chats()


def delete_chat_cb():
    st.session_state.chats.pop(st.session_state.current_chat_id, None)
    ensure_current_chat()
    st.session_state.verify_result = None
    save_chats()


def open_chat_cb(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.view = VIEWS[0]


def set_feedback(chat_id, idx, value):
    chat = st.session_state.chats.get(chat_id)

    if chat and 0 <= idx < len(chat["messages"]):
        message = chat["messages"][idx]
        # Same button parat dabla tar feedback kadhun taka
        message["feedback"] = (
            None if message.get("feedback") == value else value
        )
        if message["feedback"] is None:
            message.pop("feedback")
        save_chats()


def apply_code_cb():
    if st.session_state.ws_fixed_code:
        st.session_state.ws_code_new = st.session_state.ws_fixed_code
        st.session_state.ws_fixed_code = ""


# ==========================================
# SMALL UI PIECES
# ==========================================

def render_sources(sources):
    """Source Link Preview: link ughadnyaadhi naav, domain ani summary."""
    if not sources:
        return

    st.markdown("**🔗 Sources**")

    for i, source in enumerate(sources, start=1):
        url = source.get("url", "")
        domain = urlparse(url).netloc.replace("www.", "") or "link"
        title = (source.get("title") or domain)[:70]

        with st.expander(f"[{i}] {title} — {domain}"):
            if source.get("body"):
                st.write(source["body"])
            st.caption(f"Website: {domain}")
            st.link_button("🌐 Open source", url)


def render_feedback(chat_id, idx, message):
    feedback = message.get("feedback")

    col_up, col_down, col_text = st.columns([1, 1, 6])

    col_up.button(
        "✅ 👍" if feedback == "up" else "👍",
        key=f"fb_up_{chat_id}_{idx}",
        on_click=set_feedback,
        args=(chat_id, idx, "up"),
    )
    col_down.button(
        "✅ 👎" if feedback == "down" else "👎",
        key=f"fb_down_{chat_id}_{idx}",
        on_click=set_feedback,
        args=(chat_id, idx, "down"),
    )

    if feedback == "up":
        col_text.caption("Changle uttar mhanun mark kele.")
    elif feedback == "down":
        col_text.caption("Chukiche / kami uttar mhanun mark kele.")


# ==========================================
# VIEW: CODE WORKSPACE
# ==========================================

def run_ai_task(prompt):
    if not API_KEY:
        st.error("GEMINI_API_KEY .env file madhye set kara.")
        return None

    try:
        with st.spinner("🤖 AI vichar karat aahe..."):
            return ask_ai(prompt)
    except Exception as error:
        st.error(f"AI error: {error}")
        return None


def render_code_workspace():
    st.subheader("💻 AI Coding Workspace")
    st.caption("Code liha, chalva, errors fix kara ani samjun ghya.")

    # AI ne dilela code editor madhye bharnyasathi (widget banvaychya aadhi)
    if "ws_code_new" in st.session_state:
        st.session_state.ws_code = st.session_state.pop("ws_code_new")

    col_lang, col_exp = st.columns(2)
    ws_lang = col_lang.selectbox("Language", ["Python", "C"], key="ws_lang")
    exp_lang = col_exp.selectbox(
        "AI explanation language",
        ["Marathi", "Hindi", "English"],
        key="ws_exp_lang",
    )

    code = st.text_area("Code", key="ws_code", height=340)

    stdin_text = st.text_area(
        "Input (stdin) — optional",
        key="ws_stdin",
        height=70,
        placeholder="Jar code input() / scanf() vaparat asel tar ithe liha",
    )

    b1, b2, b3 = st.columns(3)
    run_clicked = b1.button("▶️ Run", key="ws_run", use_container_width=True)
    fix_clicked = b2.button("🛠️ Fix errors", key="ws_fix", use_container_width=True)
    explain_clicked = b3.button("📖 Explain", key="ws_explain", use_container_width=True)

    instruction = st.text_input(
        "✨ AI la sanga kay lihaycha / badlaycha",
        key="ws_instruction",
        placeholder="Example: add input validation  /  write a function for prime numbers",
    )
    edit_clicked = st.button("✨ Write / Edit with AI", key="ws_edit")

    numbered_code = "\n".join(
        f"{n}: {line}" for n, line in enumerate(code.splitlines(), start=1)
    )

    if run_clicked:
        with st.spinner("▶️ Code chalvat aahe..."):
            _, output = run_code(code, ws_lang, stdin_text)
        st.session_state.ws_output = output

    if fix_clicked:
        if not code.strip():
            st.warning("Aadhi code liha.")
        else:
            with st.spinner("▶️ Code chalvun pahat aahe..."):
                _, output = run_code(code, ws_lang, stdin_text)
            st.session_state.ws_output = output

            answer = run_ai_task(f"""
You are an expert {ws_lang} debugger.
Find all syntax, runtime and logic errors in the code below.
Use the program output / errors if they help.
Explain in {exp_lang} (keep code and technical terms in English).

Format:
1. A short list of the errors: line number, what is wrong, why.
2. The complete corrected code in ONE code block.
Do NOT include line numbers inside the corrected code.

CODE (line numbers are for reference only):
{numbered_code}

PROGRAM OUTPUT / ERRORS:
{output}
""")
            if answer:
                st.session_state.ws_ai_answer = answer
                st.session_state.ws_fixed_code = extract_code_block(answer)

    if explain_clicked:
        if not code.strip():
            st.warning("Aadhi code liha.")
        else:
            answer = run_ai_task(f"""
Explain this {ws_lang} code line by line for a beginner, in {exp_lang}
(keep code and technical terms in English).

For each line or small group of lines write:
`line number: the code` followed by a simple explanation.
Finish with a 3-line summary of what the program does and one improvement idea.

CODE:
{numbered_code}
""")
            if answer:
                st.session_state.ws_ai_answer = answer
                st.session_state.ws_fixed_code = ""

    if edit_clicked:
        if not instruction.strip():
            st.warning("AI la kay karaych te liha.")
        else:
            answer = run_ai_task(f"""
You are an expert {ws_lang} programmer.
TASK: {instruction.strip()}

Current code (may be empty):
{code}

Return the complete updated code in ONE code block (no line numbers),
then a short summary of the changes in {exp_lang}.
""")
            if answer:
                st.session_state.ws_ai_answer = answer
                st.session_state.ws_fixed_code = extract_code_block(answer)

    if st.session_state.ws_output:
        st.markdown("**📟 Output**")
        st.code(st.session_state.ws_output, language=None)

    if st.session_state.ws_ai_answer:
        st.markdown("**🤖 AI**")
        st.markdown(st.session_state.ws_ai_answer)

        if st.session_state.ws_fixed_code:
            st.button(
                "✅ Ha code editor madhye lava",
                key="ws_apply",
                on_click=apply_code_cb,
            )

    # ---------- MULTI-FILE PROJECT ----------
    st.markdown("---")
    st.subheader("📁 Multi-file project")
    st.caption("Ek project chya anek files ekatra upload karun tyabaddal vichara.")

    files = st.file_uploader(
        "Project files",
        accept_multiple_files=True,
        type=["py", "c", "h", "cpp", "js", "ts", "java", "html", "css",
              "json", "md", "txt"],
        key="ws_files",
    )

    question = st.text_area(
        "Project baddal prashna",
        key="ws_project_question",
        height=80,
        placeholder="Example: Ya project che structure samjav / Bug kuthe aahe?",
    )

    if st.button("🔍 Ask about project", key="ws_project_ask"):
        if not files:
            st.warning("Aadhi files upload kara.")
        elif not question.strip():
            st.warning("Prashna liha.")
        else:
            limit = 60000
            parts = []
            total = 0
            truncated = False

            for f in files:
                text = f.getvalue().decode("utf-8", errors="replace")
                header = f"=== FILE: {f.name} ===\n"
                room = limit - total - len(header)

                if room <= 0:
                    truncated = True
                    break

                if len(text) > room:
                    text = text[:room]
                    truncated = True

                parts.append(header + text)
                total += len(header) + len(text)

            if truncated:
                st.warning(
                    f"Project motha aahe, fakt pahile {limit} characters "
                    "AI kade geli."
                )

            answer = run_ai_task(f"""
You are a senior software engineer reviewing a multi-file project.
Answer in {exp_lang} (keep code and technical terms in English).
Mention file names (and function names) when you refer to code.
Do not invent files or code that are not shown.

PROJECT FILES:
{chr(10).join(parts)}

QUESTION:
{question.strip()}
""")
            if answer:
                st.session_state.ws_project_answer = answer

    if st.session_state.ws_project_answer:
        st.markdown("**🤖 AI**")
        st.markdown(st.session_state.ws_project_answer)


# ==========================================
# VIEW: SEARCH CHATS
# ==========================================

def render_chat_search():
    st.subheader("🔎 Search Chats")

    query = st.text_input(
        "JuNya chats madhye shodha",
        key="chat_search_query",
        placeholder="Example: python loop",
    )

    if not query.strip():
        st.info("Shabd liha, mhanje sarva chats madhun result disatil.")
        return

    hits = search_chats(query)
    st.caption(f"{len(hits)} result(s)")

    for n, (chat, idx, role, snippet) in enumerate(hits):
        with st.container(border=True):
            label = "title" if role == "title" else role
            st.markdown(
                f"**{chat['title']}** · {label} · {chat['updated'][:10]}"
            )
            st.write(f"…{snippet}…")
            st.button(
                "📂 Open chat",
                key=f"open_hit_{n}",
                on_click=open_chat_cb,
                args=(chat["id"],),
            )


# ==========================================
# VIEW: PRIVACY & DATA
# ==========================================

def wipe_chats_cb():
    st.session_state.chats = {}
    ensure_current_chat()
    st.session_state.verify_result = None
    st.session_state.last_answer = ""
    st.session_state.last_audio_output = None
    CHATS_FILE.unlink(missing_ok=True)
    LEGACY_CHAT_FILE.unlink(missing_ok=True)
    st.session_state.priv_confirm = False


def wipe_memories_cb():
    st.session_state.ai_memories = []
    MEMORY_FILE.unlink(missing_ok=True)
    st.session_state.priv_confirm = False


def wipe_all_cb():
    wipe_chats_cb()
    wipe_memories_cb()
    st.session_state.galaxy_pdf_text = ""
    st.session_state.galaxy_pdf_pages_list = []
    st.session_state.galaxy_pdf_name = ""
    st.session_state.galaxy_pdf_pages = 0
    st.session_state.pdf_uploader_key += 1
    st.session_state.generated_image = None
    st.session_state.priv_confirm = False


def render_privacy():
    st.subheader("🔒 Privacy & Data Controls")
    settings = st.session_state.settings

    st.markdown("##### Settings")

    save_history = st.toggle(
        "💾 Chat history disk var save kara",
        value=settings["save_history"],
        key="priv_save_history",
        help="OFF kelyas navin chats fakt session paryant rahatil.",
    )
    use_memory = st.toggle(
        "🧠 AI la mazhi memory vaparu dya",
        value=settings["use_memory"],
        key="priv_use_memory",
        help="OFF kelyas saved memories AI kade jaat nahit.",
    )

    day_options = [0, 7, 30, 90, 365]
    current_days = settings.get("auto_delete_days", 0)
    auto_days = st.selectbox(
        "🧹 Juni chats auto-delete",
        day_options,
        index=day_options.index(current_days) if current_days in day_options else 0,
        format_func=lambda d: "Never" if d == 0 else f"{d} divasanpeksha juni",
        key="priv_days",
    )

    changed = (
        save_history != settings["save_history"]
        or use_memory != settings["use_memory"]
        or auto_days != settings.get("auto_delete_days", 0)
    )

    if changed:
        settings["save_history"] = save_history
        settings["use_memory"] = use_memory
        settings["auto_delete_days"] = auto_days
        save_settings()
        if save_history:
            save_chats()
        st.success("Settings save zhalya.")

    if auto_days and st.button("🧹 Auto-delete aata laga", key="priv_purge_now"):
        removed = purge_old_chats(
            auto_days, keep_id=st.session_state.current_chat_id
        )
        save_chats()
        st.success(f"{removed} juni chat(s) delete zhalya.")

    st.caption(
        "Incognito chat: sidebar madhun chat var 🕶️ Incognito select kara — "
        "ti disk var save honar nahi."
    )

    st.markdown("---")
    st.markdown("##### Tumcha data")

    all_messages = [
        m for c in st.session_state.chats.values() for m in c["messages"]
    ]
    ups = sum(1 for m in all_messages if m.get("feedback") == "up")
    downs = sum(1 for m in all_messages if m.get("feedback") == "down")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chats", len(st.session_state.chats))
    m2.metric("Messages", len(all_messages))
    m3.metric("Memories", len(st.session_state.ai_memories))
    m4.metric("👍 / 👎", f"{ups} / {downs}")

    st.caption(f"Data folder: {BASE_DIR}")

    export_all = json.dumps(
        {
            "chats": st.session_state.chats,
            "memories": st.session_state.ai_memories,
            "settings": st.session_state.settings,
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "⬇️ Maza sarva data download kara (JSON)",
        data=export_all,
        file_name="galaxy_ai_all_data.json",
        mime="application/json",
        key="priv_download_all",
    )

    st.markdown("---")
    st.markdown("##### ⚠️ Danger zone")

    confirm = st.checkbox("Mala khatri aahe (delete parat milat nahi)", key="priv_confirm")

    d1, d2, d3 = st.columns(3)
    d1.button(
        "🗑️ Delete all chats",
        key="priv_wipe_chats",
        on_click=wipe_chats_cb,
        disabled=not confirm,
        use_container_width=True,
    )
    d2.button(
        "🗑️ Delete all memories",
        key="priv_wipe_memories",
        on_click=wipe_memories_cb,
        disabled=not confirm,
        use_container_width=True,
    )
    d3.button(
        "🔥 Delete everything",
        key="priv_wipe_all",
        on_click=wipe_all_cb,
        disabled=not confirm,
        use_container_width=True,
    )



# ==========================================
# SIDEBAR
# ==========================================

attached_image = None  # (bytes, mime) - pudhchya message sobat jail

with st.sidebar:
    st.markdown("# 🌌 Galaxy AI")
    st.caption("Your personal AI assistant")

    view = st.radio(
        "View",
        VIEWS,
        key="view",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ---------- CHATS ----------
    ensure_current_chat()
    st.subheader("💬 Chats")

    st.button(
        "➕ New Chat",
        key="new_chat_btn",
        on_click=new_chat_cb,
        use_container_width=True,
        type="primary",
    )

    _all_chats = sorted(
        st.session_state.chats.values(),
        key=lambda c: c["updated"],
        reverse=True,
    )
    _used_chats = sum(1 for c in _all_chats if c["messages"])
    _total_msgs = sum(len(c["messages"]) for c in _all_chats)

    st.caption(f"📊 Ata paryant chats: {_used_chats} · Messages: {_total_msgs}")

    def _chat_button(chat):
        is_current = chat["id"] == st.session_state.current_chat_id
        label = (
            ("🟢 " if is_current else "💬 ")
            + ("🕶️ " if chat.get("incognito") else "")
            + chat["title"][:28]
        )
        st.button(
            label,
            key=f"chat_{'current' if is_current else 'pick'}_{chat['id']}",
            on_click=open_chat_cb,
            args=(chat["id"],),
            use_container_width=True,
            help=f"{len(chat['messages'])} messages · {chat['updated'][:10]}",
        )

    for _chat in _all_chats[:8]:
        _chat_button(_chat)

    if len(_all_chats) > 8:
        with st.expander(f"📚 Juni chats ({len(_all_chats) - 8})"):
            for _chat in _all_chats[8:]:
                _chat_button(_chat)

    cur = current_chat()

    nc1, nc2 = st.columns(2)
    nc1.button("🧹 Clear", key="clear_chat_btn", on_click=clear_chat_cb,
               help="Hi chat rikami kara", use_container_width=True)
    nc2.button("🗑️ Delete", key="delete_chat_btn", on_click=delete_chat_cb,
               help="Hi chat delete kara", use_container_width=True)

    # Key madhye title aahe, mhanun title badalla ki widget navin hoto
    new_title = st.text_input(
        "✏️ Chat che naav",
        value=cur["title"],
        key=f"rename_{cur['id']}_{cur['title']}",
    )
    if new_title.strip() and new_title.strip() != cur["title"]:
        cur["title"] = new_title.strip()[:80]
        save_chats()
        st.rerun()

    incognito = st.checkbox(
        "🕶️ Incognito (disk var save honar nahi)",
        value=bool(cur.get("incognito", False)),
        key=f"incog_{cur['id']}",
    )
    if incognito != bool(cur.get("incognito", False)):
        cur["incognito"] = incognito
        save_chats()

    st.markdown("---")

    # ---------- PDF READER ----------
    with st.expander("📄 PDF Reader", expanded=bool(st.session_state.galaxy_pdf_name)):
        uploaded_pdf = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            key=f"galaxy_pdf_upload_{st.session_state.pdf_uploader_key}",
        )

        if uploaded_pdf is not None:
            pdf_id = f"{uploaded_pdf.name}-{uploaded_pdf.size}"

            # Read the PDF only once per new file
            if pdf_id != st.session_state.galaxy_pdf_name:
                try:
                    reader = PdfReader(uploaded_pdf)
                    pages_text = [
                        page.extract_text() or "" for page in reader.pages
                    ]

                    st.session_state.galaxy_pdf_pages_list = pages_text
                    st.session_state.galaxy_pdf_text = "\n".join(pages_text)
                    st.session_state.galaxy_pdf_pages = len(pages_text)
                    st.session_state.galaxy_pdf_name = pdf_id

                except Exception as error:
                    st.session_state.galaxy_pdf_pages_list = []
                    st.session_state.galaxy_pdf_text = ""
                    st.session_state.galaxy_pdf_name = ""
                    st.error(f"PDF read error: {error}")

        if st.session_state.galaxy_pdf_name:
            text_len = len(st.session_state.galaxy_pdf_text.strip())

            if text_len:
                st.success(
                    f"PDF loaded! Pages: {st.session_state.galaxy_pdf_pages}, "
                    f"Characters: {text_len}"
                )
                if text_len > PDF_CHAR_LIMIT:
                    st.caption(
                        "PDF motha aahe: tumchya prashnashi jupnare "
                        "bhag (page number sahit) AI kade jatil."
                    )
                st.caption("Ata chat madhye PDF baddal prashna vichara.")
            else:
                st.warning(
                    "Ya PDF madhye text milala nahi (scanned/image PDF asu shakto)."
                )

            if st.button("🗑️ Remove PDF", key="remove_galaxy_pdf"):
                st.session_state.galaxy_pdf_text = ""
                st.session_state.galaxy_pdf_pages_list = []
                st.session_state.galaxy_pdf_name = ""
                st.session_state.galaxy_pdf_pages = 0
                st.session_state.pdf_uploader_key += 1  # uploader reset
                st.rerun()

    # ---------- IMAGE UNDERSTANDING ----------
    with st.expander("🖼️ Image Understanding"):
        uploaded_image = st.file_uploader(
            "Photo kiwa screenshot",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"img_upload_{st.session_state.img_uploader_key}",
        )

        if uploaded_image is not None:
            attached_image = (
                uploaded_image.getvalue(),
                uploaded_image.type or "image/png",
            )
            st.image(uploaded_image, use_container_width=True)
            st.caption(
                "Hi image pudhchya message sobat AI kade jail. "
                "Example: 'Hya screenshot madhil error samjav'."
            )

    st.markdown("---")

    # ---------- AI MEMORY ----------
    st.subheader("🧠 AI Memory")

    memory_text = st.text_input(
        "Kay lakshat thevayche?",
        key="memory_text_input",
        placeholder="Example: I am learning Python",
    )

    if st.button("💾 Save Memory", key="save_memory_button"):
        new_memory = memory_text.strip()

        if new_memory:
            if new_memory not in st.session_state.ai_memories:
                st.session_state.ai_memories.append(new_memory)
                save_memories()
                st.success("Memory saved!")
                st.rerun()
            else:
                st.info("Hi memory aadhipasun save aahe.")

    with st.expander("📚 Saved Memories"):
        for i, memory in enumerate(st.session_state.ai_memories):
            col1, col2 = st.columns([4, 1])

            col1.write(memory)

            if col2.button("🗑️", key=f"delete_memory_{i}"):
                st.session_state.ai_memories.pop(i)
                save_memories()
                st.rerun()

    st.markdown("---")

    # ---------- SETTINGS ----------
    language = st.selectbox("Reply language", LANGUAGES, index=0)

    topic = st.selectbox("AI mode", AI_MODES)

    use_web_search = st.toggle(
        "🌐 Live Web Search",
        value=False,
        key="use_web_search",
        help="ON: internet var shodhun uttar dil jail ani source links disatil.",
    )

    with st.expander("🎓 AI Learning Mode"):
        learning_on = st.toggle(
            "Learning mode ON",
            value=False,
            key="learning_on",
            help="AI vishay beginner pasun step-by-step shikvel.",
        )
        learning_level = st.selectbox(
            "Level",
            ["Beginner", "Intermediate", "Advanced"],
            key="learning_level",
        )

    st.markdown("---")

    # ---------- IMAGE CREATOR ----------
    with st.expander("🎨 AI Image Creator"):
        image_prompt = st.text_area(
            "Describe the image you want",
            placeholder="Example: A beautiful galaxy in space",
            key="image_prompt",
        )

        if st.button("🎨 Generate Image", key="generate_image"):
            if not image_prompt.strip():
                st.warning("Please enter an image description.")
            elif not API_KEY:
                st.error("Please configure GEMINI_API_KEY in your .env file.")
            else:
                try:
                    client = get_gemini_client(API_KEY)

                    with st.spinner("Creating your image..."):
                        response = client.models.generate_content(
                            model="gemini-2.5-flash-image",
                            contents=image_prompt.strip(),
                            config=types.GenerateContentConfig(
                                response_modalities=["IMAGE"]
                            ),
                        )

                    image_bytes = None

                    for candidate in response.candidates or []:
                        if not candidate.content:
                            continue
                        for part in candidate.content.parts or []:
                            if part.inline_data is not None:
                                image_bytes = part.inline_data.data
                                break
                        if image_bytes:
                            break

                    if image_bytes:
                        st.session_state.generated_image = image_bytes
                    else:
                        st.session_state.generated_image = None
                        st.warning("No image was returned. Try another prompt.")

                except Exception as error:
                    st.session_state.generated_image = None
                    st.error(f"Image generation failed: {error}")

        if st.session_state.generated_image:
            st.image(
                st.session_state.generated_image,
                caption="Generated by Galaxy AI",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download Image",
                data=st.session_state.generated_image,
                file_name="galaxy_ai_image.png",
                mime="image/png",
                key="download_generated_image",
            )

    # ---------- EXPORT ----------
    with st.expander("💾 Export this chat"):
        export_format = st.selectbox(
            "Format",
            ["TXT", "Markdown", "HTML", "JSON", "PDF"],
            key="export_format",
        )

        safe_name = re.sub(r"[^\w\-]+", "_", cur["title"]).strip("_")[:30] or "chat"

        export_data = None
        export_mime = "text/plain"
        export_ext = "txt"

        if export_format == "TXT":
            export_data = chat_to_text(cur)
        elif export_format == "Markdown":
            export_data = chat_to_markdown(cur)
            export_mime, export_ext = "text/markdown", "md"
        elif export_format == "HTML":
            export_data = chat_to_html(cur)
            export_mime, export_ext = "text/html", "html"
        elif export_format == "JSON":
            export_data = json.dumps(cur, ensure_ascii=False, indent=2)
            export_mime, export_ext = "application/json", "json"
        else:
            export_data = chat_to_pdf(cur)
            export_mime, export_ext = "application/pdf", "pdf"

            if export_data is None:
                st.warning(
                    "PDF sathi `pip install reportlab` kara. "
                    "Marathi chya perfect PDF sathi HTML export kara ani "
                    "browser madhun Print -> Save as PDF vapra."
                )
            else:
                st.caption(
                    "Tip: Marathi (Devanagari) madhe jodakshare PDF madhye "
                    "thodi chukichi disu shaktat. HTML export jast changla aahe."
                )

        if export_data is not None:
            st.download_button(
                f"⬇️ Download {export_format}",
                data=export_data,
                file_name=f"{safe_name}.{export_ext}",
                mime=export_mime,
                key="download_chat_export",
                use_container_width=True,
            )

    st.markdown("---")

    current_model = st.session_state.get("working_model")
    if current_model:
        st.caption(f"Active model: {current_model}")

    st.caption("✨ Galaxy AI")


# ==========================================
# MAIN PAGE
# ==========================================

st.title("🌌 Galaxy AI Chat")
st.markdown("##### ✨ Ask anything. Learn anything. Build anything.")
st.markdown("---")

if view == VIEWS[1]:
    render_code_workspace()
    st.stop()

if view == VIEWS[2]:
    render_chat_search()
    st.stop()

if view == VIEWS[3]:
    render_privacy()
    st.stop()

# ---------- CHAT VIEW ----------
cur = current_chat()
msgs = cur["messages"]

# Layout:
#   - Chat messages madhye
#   - Voice box (mic + options) screen chya khali FIXED (CSS ne)
#   - Typing box (st.chat_input) tyachya khali FIXED
input_area = st.container(key="voice_bar")
chat_area = st.container()


# ==========================================
# INPUT AREA (TOP)
# ==========================================

prompt = ""
voice_audio = None
voice_mime = "audio/wav"
answer_with_voice = False

with input_area:
    opt1, opt2, opt3, opt4 = st.columns([1.3, 1.3, 1.1, 2.6], gap="small")

    with opt1:
        voice_language = st.selectbox(
            "Voice input language",
            VOICE_INPUT_LANGUAGES,
            key="voice_language",
            label_visibility="collapsed",
            format_func=lambda x: f"🎤 {x}",
        )

    with opt2:
        voice_answer_mode = st.selectbox(
            "Voice answer",
            [
                "Auto (mic ne bollyas)",
                "Always ON",
                "OFF",
            ],
            key="voice_answer_mode",
            label_visibility="collapsed",
            format_func=lambda x: {
                "Auto (mic ne bollyas)": "🔊 Auto",
                "Always ON": "🔊 ON",
                "OFF": "🔇 OFF",
            }[x],
            help="Voice answer: Auto = mic ne bollyas, ON = nehmi, OFF = kadhich nahi",
        )

    with opt3:
        instant_voice = st.toggle(
            "⚡ Instant",
            value=False,
            key="instant_voice",
            help=(
                "OFF: mic cha text aadhi disel, durust karun send kara. "
                "ON: audio direct AI kade jail."
            ),
        )

    with opt4:
        audio_clip = st.audio_input(
            "🎤 Mic",
            key="galaxy_mic",
            label_visibility="collapsed",
        )

    if audio_clip is not None:
        audio_bytes = audio_clip.getvalue()
        audio_mime = audio_clip.type or "audio/wav"
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash

            if not API_KEY:
                st.error("GEMINI_API_KEY .env file madhye set kara.")

            elif instant_voice:
                voice_audio = audio_bytes
                voice_mime = audio_mime
                answer_with_voice = True

            else:
                with st.spinner("🎧 Voice text madhye convert hot aahe..."):
                    try:
                        recognized_text = transcribe_audio(
                            audio_bytes,
                            voice_language,
                            audio_mime,
                        )

                        if recognized_text:
                            st.session_state.voice_draft = recognized_text
                            st.session_state.voice_draft_active = True
                        else:
                            st.warning("Clear speech milala nahi. Punha bola.")

                    except Exception as error:
                        st.error(f"Voice transcription error: {error}")

    # Voice draft (multi-line editable box)
    if st.session_state.voice_draft_active:
        st.text_area(
            "🎧 Olkhlela text tapasa kiwa durust kara",
            key="voice_draft",
            height=68,
        )

        st.button(
            "🔊 Send voice message",
            on_click=send_voice_draft,
            use_container_width=True,
        )

    if st.session_state.pending_voice_prompt:
        prompt = st.session_state.pending_voice_prompt
        st.session_state.pending_voice_prompt = ""
        answer_with_voice = True


# ==========================================
# TYPING BOX (FIXED AT BOTTOM)
# ==========================================
# st.chat_input container chya BAHER (top level) pahije,
# tarach to screen chya khali fixed rahto.
# Enter = send, Shift+Enter = new line

typed_prompt = st.chat_input(
    "✨ Type your message... (Enter = Send, Shift+Enter = new line)",
    key="galaxy_chat_input",
)

if (
    typed_prompt
    and typed_prompt.strip()
    and not prompt
    and voice_audio is None
):
    prompt = typed_prompt.strip()
    answer_with_voice = False


# ==========================================
# VOICE ANSWER MODE
# ==========================================

if voice_answer_mode == "Always ON":
    speak_answer = True
elif voice_answer_mode == "OFF":
    speak_answer = False
else:
    speak_answer = answer_with_voice


# ==========================================
# CHAT AREA (BELOW INPUT)
# ==========================================

with chat_area:
    st.markdown("---")

    chat_id = cur["id"]

    # Display old messages (with sources + feedback buttons)
    for idx, message in enumerate(msgs):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources"))
                render_feedback(chat_id, idx, message)

    # Generate answer
    if prompt or voice_audio is not None:
        if prompt:
            user_content = (
                f"🖼️ *(image attached)*\n\n{prompt}"
                if attached_image
                else prompt
            )

            msgs.append({"role": "user", "content": user_content})

            if cur["title"] == "New chat":
                cur["title"] = prompt[:40]
            cur["updated"] = now_iso()

            with st.chat_message("user"):
                st.markdown(user_content)

            # Save immediately, so the user message isn't lost on error.
            save_chats()

        else:
            with st.chat_message("user"):
                st.markdown("🎤 Audio received...")

        answer = ""
        error_text = ""
        no_speech = False
        heard = ""
        sources = []
        answered = False

        with st.chat_message("assistant"):
            if not API_KEY:
                error_text = (
                    "GEMINI_API_KEY milali nahi. "
                    "Project folder madhil .env file tapasa."
                )

            else:
                try:
                    # Live web search (fakt typed / voice-draft prompt sathi)
                    if use_web_search and prompt:
                        with st.spinner("🌐 Internet var shodhat aahe..."):
                            sources = web_search(prompt)

                    system_prompt = build_system_prompt(
                        language,
                        topic,
                        prompt,
                        learning_on,
                        learning_level,
                    )

                    if sources:
                        system_prompt += (
                            "\nWEB SEARCH RESULTS. Use them when relevant and "
                            "cite them as [1], [2] ... in your answer:\n"
                            + format_search_results(sources)
                            + "\n"
                        )

                    if attached_image:
                        system_prompt += (
                            "\nThe user attached an image to this message. "
                            "Look at it carefully (text, objects, screenshots, "
                            "errors, charts) and answer about it.\n"
                        )

                    # Build history without duplicating the latest typed prompt.
                    previous_messages = msgs[:-1] if prompt else msgs

                    history = "\n\n".join(
                        f"{m['role'].upper()}: {m['content']}"
                        for m in previous_messages[-10:]
                    )

                    if voice_audio is not None:
                        contents = [
                            system_prompt
                            + """
The latest user message is the attached audio.
Return exactly this format:
HEARD: <transcription of the user's words>
REPLY: <answer to the user's words>
Do not translate the transcription.
If there is no clear speech, return:
HEARD: NONE
REPLY: NONE

Conversation so far:
"""
                            + history,
                            types.Part.from_bytes(
                                data=voice_audio,
                                mime_type=voice_mime,
                            ),
                        ]

                    else:
                        contents = [
                            system_prompt
                            + "\n\nConversation so far:\n"
                            + history
                            + "\n\nLATEST USER MESSAGE:\n"
                            + prompt
                        ]

                    if attached_image:
                        contents.append(
                            types.Part.from_bytes(
                                data=attached_image[0],
                                mime_type=attached_image[1],
                            )
                        )

                    with st.spinner("✨ Galaxy AI vichar karat aahe..."):
                        response = gemini_generate(contents)

                    raw_text = (response.text or "").strip()

                    if voice_audio is not None:
                        match = re.search(
                            r"HEARD:\s*(.*?)\s*REPLY:\s*(.*)",
                            raw_text,
                            flags=re.S | re.I,
                        )

                        if match:
                            heard = match.group(1).strip()
                            answer = match.group(2).strip()
                        else:
                            # Format milala nahi -> poorna text uttar mana
                            answer = raw_text

                        if (
                            heard.upper() == "NONE"
                            or answer.upper() == "NONE"
                            or (match and not heard)
                        ):
                            no_speech = True
                            answer = ""
                    else:
                        answer = raw_text

                except Exception as error:
                    error_text = str(error)

            # ----- Show result + save -----
            if error_text:
                st.error(error_text)

            elif no_speech:
                st.warning("Clear speech milala nahi. Punha bola.")

            elif not answer:
                st.warning("AI ne rikama uttar dila. Punha prayatna kara.")

            else:
                if heard:
                    st.caption(f"🎧 Heard: {heard}")

                st.markdown(answer)

                if voice_audio is not None and heard:
                    msgs.append({"role": "user", "content": heard})
                    if cur["title"] == "New chat":
                        cur["title"] = heard[:40]

                assistant_message = {"role": "assistant", "content": answer}
                if sources:
                    assistant_message["sources"] = sources

                msgs.append(assistant_message)
                cur["updated"] = now_iso()
                st.session_state.last_answer = answer
                st.session_state.verify_result = None
                save_chats()
                answered = True

                render_sources(sources)
                render_feedback(chat_id, len(msgs) - 1, assistant_message)

        # ----- Image fakt ekach message sathi -----
        if answered and attached_image:
            st.session_state.img_uploader_key += 1

        # ----- Voice answer -----
        if answered and speak_answer:
            try:
                with st.spinner("🔊 Awaaz tayar hot aahe..."):
                    audio_out = make_speech(answer, language)

                if audio_out:
                    st.session_state.last_audio_output = audio_out
                    st.audio(audio_out, format="audio/mp3", autoplay=True)
            except Exception as error:
                st.warning(f"Voice answer tayar karta aala nahi: {error}")

    # ---------- SOURCE VERIFICATION ----------
    if msgs and msgs[-1]["role"] == "assistant":
        if st.button("✅ Verify last answer", key="verify_btn",
                     help="Uttar internet var chya sources shi tapasa"):
            if not API_KEY:
                st.error("GEMINI_API_KEY .env file madhye set kara.")
            else:
                question = next(
                    (m["content"] for m in reversed(msgs[:-1])
                     if m["role"] == "user"),
                    "",
                )
                try:
                    with st.spinner("🔎 Sources tapasat aahe..."):
                        verdict, verify_sources = verify_answer(
                            question, msgs[-1]["content"]
                        )

                    st.session_state.verify_result = {
                        "chat_id": chat_id,
                        "idx": len(msgs) - 1,
                        "text": verdict,
                        "sources": verify_sources,
                    }
                except Exception as error:
                    st.error(f"Verification failed: {error}")

        verify = st.session_state.verify_result

        if (
            verify
            and verify["chat_id"] == chat_id
            and verify["idx"] == len(msgs) - 1
        ):
            with st.container(border=True):
                st.markdown("### ✅ Source Verification")
                st.markdown(verify["text"])
                render_sources(verify["sources"])

    # Replay button (fakt jevha audio uplabdh asel)
    if st.session_state.last_audio_output:
        if st.button("🔁 Replay last voice answer", key="replay_audio"):
            st.audio(
                st.session_state.last_audio_output,
                format="audio/mp3",
            )
