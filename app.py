"""
app.py — Streamlit UI for MIT CS Catalog RAG Assistant
Fixed: proper session state for input persistence + response display
Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # fallback for runtime

import streamlit as st
from src import MITCatalogAssistant

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIT CS Course Planning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #a31f34 0%, #8b1a2b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #f8c8ce; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .answer-box {
        background: #f0f7ff;
        border: 1px solid #b8d4f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
    }
    .question-label {
        font-weight: 600;
        color: #a31f34;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "assistant" not in st.session_state:
    st.session_state.assistant = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 MIT CS Course Planning Assistant</h1>
    <p>Grounded in the MIT Course 6-3 (CS&amp;E) Academic Catalog &nbsp;·&nbsp;
       Powered by LangChain + Groq (Llama 3) &nbsp;·&nbsp;
       Every answer cites its source</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "Can I take 6.1210 if I've done 6.1010 and 6.1200 with a C?",
        "What do I need before enrolling in 6.3900 (Intro to ML)?",
        "I completed 6.3900. What do I need for 6.3940 (Deep Learning)?",
        "What is the full prereq chain to reach 6.3940 from scratch?",
        "How many AUS subjects do I need and what are the group rules?",
        "Is my AUS combo valid: 6.3940 (C), 6.4110 (C), 6.5840 (B)?",
        "What happens if I get a D in a Foundation subject?",
        "What is the max units per semester without special approval?",
        "Can 6.3900 and 18.06 be taken at the same time?",
        "Can transfer credits satisfy Foundation subjects?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=True):
            st.session_state.input_text = ex

    st.markdown("---")
    st.markdown("### 📚 Sources")
    st.markdown("""
- [MIT Course 6 Catalog](https://catalog.mit.edu/subjects/6/)
- [Course 6-3 Degree Chart](https://catalog.mit.edu/degree-charts/computer-science-engineering-course-6-3/)
- [Academic Policies](https://catalog.mit.edu/mit/undergraduate-education/academic-policies/)
- [EECS Advising FAQ](https://www.eecs.mit.edu/academics-admissions/undergraduate-programs/course-6-3-cs/faq/)
""")

# ── Input Area ────────────────────────────────────────────────────────────────
st.markdown("#### Ask a question about MIT CS courses or prerequisites")

question = st.text_area(
    label="Your question",
    value=st.session_state.input_text,
    height=100,
    placeholder=(
        "e.g. 'I've completed 6.1010 and 6.1200 with a B. "
        "Can I take 6.1210 next semester?'"
    ),
    key="question_input",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    ask_clicked = st.button("🔍 Ask Assistant", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.input_text = ""
        st.rerun()

# ── Handle Ask ────────────────────────────────────────────────────────────────
if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    elif not api_key.strip():
        st.error("⚠️ Enter your Groq API key in the sidebar. Get one free at https://console.groq.com")
    else:
        # Load or reuse assistant
        if st.session_state.assistant is None:
            with st.spinner("Loading catalog index..."):
                try:
                    st.session_state.assistant = MITCatalogAssistant(groq_api_key=api_key)
                except FileNotFoundError as e:
                    st.error(f"❌ {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Failed to load assistant: {e}")
                    st.stop()

        with st.spinner("🔎 Searching catalog and generating answer..."):
            try:
                answer = st.session_state.assistant.ask(question)
                # Save to history
                st.session_state.history.append({
                    "q": question,
                    "a": answer,
                })
                # Clear input after successful response
                st.session_state.input_text = ""
            except Exception as e:
                st.error(f"❌ Error from Groq: {e}")
                st.stop()

        st.rerun()

# ── Display History ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Conversation History")
    for i, turn in enumerate(reversed(st.session_state.history)):
        with st.container():
            st.markdown(
                f'<div class="question-label">❓ Question {len(st.session_state.history) - i}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f"**{turn['q']}**")
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(turn["a"])
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("👆 Type a question above and click **Ask Assistant** to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ This assistant is grounded in catalog documents only. "
    "Always verify with your academic advisor and the MIT Registrar. "
    "Subject availability may change each semester."
)