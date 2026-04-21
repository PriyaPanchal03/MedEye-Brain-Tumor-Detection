
#----New--
import streamlit as st
from chatbot import is_greeting, medical_chatbot

st.set_page_config(layout="wide")

def render_chatbot():

    # ---------- STYLE ----------
    st.markdown("""
    <style>
    
    .chat-wrapper {
        width: 100%;
        border-radius: 16px;
        background: #ffffff;
        border: 2px solid #cfd8ea;
        box-shadow: 0px 10px 30px rgba(31,43,77,0.15);
        overflow: hidden;
    }
    .chat-wrapper:hover {
        box-shadow: 0 14px 35px rgba(31, 43, 77, 0.22);
    }
    .chat-header {
        background: #1f2b4d;
        color: white;
        padding: 14px;
        font-size: 25px;
        font-weight: 650;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }

    .chat-body {
        height: 300px;
        padding: 12px;
        overflow-y: auto;
        border-top: 1px solid #d6dbea;
        background: #f9fbff;
    }

    .user-msg {
        background: #4f6ef7;
        color: white;
        padding: 10px 14px;
        border-radius: 18px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 10px;
        font-size: 14px;
    }

    .bot-msg {
        background: white;
        padding: 10px 14px;
        border-radius: 18px;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 10px;
        font-size: 14px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
     # Cache & limits
    if "api_calls" not in st.session_state:
        st.session_state.api_calls = 0
   
    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}

    # ---------- WIDTH CONTROL----------
    left, center, right = st.columns([1, 2, 1])

    with center:   # 
        st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
        st.markdown("<div class='chat-header'>🤖 MedBot-AI Assistant</div>", unsafe_allow_html=True)
        st.markdown("<div class='chat-body'>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("", placeholder="Ask about tumor, MRI...")
            send = st.form_submit_button("Send")

        st.markdown("</div>", unsafe_allow_html=True)

        if send and user_question.strip():

            # Save user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            if is_greeting(user_question):
                answer = "Hello 👋 I can help with questions about MRI scans, tumors, and general brain health."
            elif st.session_state.api_calls >= 20:
                answer = "⚠️ Daily AI limit reached. Please try again later."
            else:
                if user_question in st.session_state.qa_cache:
                    answer = st.session_state.qa_cache[user_question]
                else:
                    with st.spinner("Thinking..."):
                        answer = medical_chatbot(user_question)
                        st.session_state.qa_cache[user_question] = answer
                        st.session_state.api_calls += 1

            # Save bot reply
            st.session_state.chat_history.append({
                "role": "bot",
                "content": answer
            })

            #  Force immediate rerun
            st.rerun()
