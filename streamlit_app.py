import os
import streamlit as st
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# Import the bot and state from our backend file
from langraph_chatbot import chatbot, ChatState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

# 1. Setup & Environment
load_dotenv()

# 2. Streamlit UI Configuration
st.set_page_config(page_title="Stock HITL Bot", page_icon="📈", layout="centered")

# Custom CSS for a better look
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Stock Trading Bot")
st.markdown("### Professional Human-In-The-Loop Trading Assistant")
st.info("Ask me for stock prices or request to buy shares. I will ask for your approval before any purchase!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Unique thread per session
    st.session_state.thread_id = "streamlit-user-session"
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Sync & Display History ---
# Always fetch the most recent state from the LangGraph backend
state = chatbot.get_state(config)
graph_messages = state.values.get("messages", [])

# Convert LangGraph messages to a format Streamlit understands
display_messages = []
for msg in graph_messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    display_messages.append({"role": role, "content": msg.content})

# Render the chat history
for msg in display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handling Human Approval (HITL) ---
if st.session_state.awaiting_approval:
    with st.chat_message("assistant"):
        st.warning(f"**Action Required:** {st.session_state.awaiting_approval}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Approve"):
                # Resume the graph with "yes"
                result = chatbot.invoke(Command(resume="yes"), config=config)
                st.session_state.awaiting_approval = None
                st.rerun()
        with col2:
            if st.button("❌ No, Decline"):
                # Resume the graph with "no"
                result = chatbot.invoke(Command(resume="no"), config=config)
                st.session_state.awaiting_approval = None
                st.rerun()

# --- User Input ---
if prompt := st.chat_input("Example: 'Buy 10 shares of AAPL' or 'Price of TSLA'"):
    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process with LangGraph backend
    result = chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # 3. Check for interrupts (HITL)
    snapshot = chatbot.get_state(config)
    if snapshot.next:
        for task in snapshot.tasks:
            if task.interrupts:
                st.session_state.awaiting_approval = task.interrupts[0].value
                break

    # 4. If no interrupt, the result is already in the state,
    # the next rerun will display it via the sync logic at the top.
    st.rerun()
