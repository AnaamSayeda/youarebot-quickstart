# app/web/streamlit_app.py

import os
import requests
import streamlit as st
from uuid import uuid4
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("# Echo bot 🚀 + Live Metrics")
st.sidebar.markdown("# Echo bot 🚀 + Live Metrics")

# Session state
if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything!"}]
if "probs" not in st.session_state:
    st.session_state.probs = []
if "labels" not in st.session_state:
    st.session_state.labels = []

# Config
default_url = "http://localhost:6872"

with st.sidebar:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

    st.text_input("Orchestrator URL", key="echo_bot_url", value=default_url, disabled=True)
    st.text_input("Dialog ID", key="dialog_id", disabled=True)

# Show history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
if message := st.chat_input("Your message"):
    user_msg = {"role": "user", "content": message}
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(message)

    try:
        # Classifier call
        pred_resp = requests.post(
            f"{default_url}/predict",
            json={
                "id": str(uuid4()),
                "dialog_id": st.session_state.dialog_id,
                "participant_index": 0,
                "text": message
            }
        ).json()
        user_prob = pred_resp["is_bot_probability"]
        st.session_state.probs.append(user_prob)
        st.session_state.labels.append(0)
        st.write(f"🤖 BOT probability (you): {user_prob:.2f}")
    except Exception as e:
        st.error(f"Classifier call failed: {e}")

    try:
        # LLM call
        chat_resp = requests.post(
            f"{default_url}/get_message",
            json={
                "dialog_id": st.session_state.dialog_id,
                "last_msg_id": str(uuid4()),
                "last_msg_text": message
            }
        ).json()
        bot_text = chat_resp.get("choices", [{}])[0].get("message", {}).get("content", "🤖 No response")
        bot_msg = {"role": "assistant", "content": bot_text}
        st.session_state.messages.append(bot_msg)
        st.chat_message("assistant").write(bot_text)

        # Classify bot response
        pred_resp_bot = requests.post(
            f"{default_url}/predict",
            json={
                "id": str(uuid4()),
                "dialog_id": st.session_state.dialog_id,
                "participant_index": 1,
                "text": bot_text
            }
        ).json()
        bot_prob = pred_resp_bot["is_bot_probability"]
        st.session_state.probs.append(bot_prob)
        st.session_state.labels.append(1)
        st.write(f"🤖 BOT probability (LLM): {bot_prob:.2f}")
    except Exception as e:
        st.error(f"LLM call failed: {e}")

    st.rerun()

# Metrics
if len(st.session_state.probs) > 0:
    preds = [1 if p >= 0.5 else 0 for p in st.session_state.probs]
    acc = accuracy_score(st.session_state.labels, preds)
    try:
        ll = log_loss(st.session_state.labels, st.session_state.probs)
    except ValueError:
        ll = np.nan

    st.sidebar.markdown(f"**Live Accuracy:** {acc:.2f}")
    st.sidebar.markdown(f"**Live Log-Loss:** {ll:.2f}")

    acc_progress, ll_progress = [], []
    for i in range(1, len(st.session_state.probs) + 1):
        acc_progress.append(
            accuracy_score(st.session_state.labels[:i], [1 if p >= 0.5 else 0 for p in st.session_state.probs[:i]])
        )
        try:
            ll_progress.append(log_loss(st.session_state.labels[:i], st.session_state.probs[:i]))
        except ValueError:
            ll_progress.append(np.nan)

    st.line_chart({"Accuracy": acc_progress, "LogLoss": ll_progress})
