# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground/main.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from utils import GATEWAY_CONFIG, MODEL_CHOICE, UnifiedGatewayLLM, resolve_model_string  # noqa: E402

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL = "gpt-6@nvidia"


@st.cache_resource
def get_llm() -> UnifiedGatewayLLM:
    return UnifiedGatewayLLM(GATEWAY_CONFIG, num_concurrency=1, num_max_retry=2, timeout=600)


st.set_page_config(page_title="Model Playground", layout="wide")
st.title("Model Playground")

with st.sidebar:
    model_options = list(MODEL_CHOICE.keys())
    model_name = st.selectbox("Model", model_options, index=model_options.index(DEFAULT_MODEL))
    # gpt-6-astra rejects the temperature parameter, so it is opt-in.
    send_temperature = st.checkbox("Send temperature", value=False)
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05, disabled=not send_temperature)
    max_tokens = st.number_input("Max tokens", min_value=256, max_value=131072, value=16384, step=1024)
    system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=150)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.caption(f"Resolved model string: `{resolve_model_string(model_name)}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    request = {
        "model": resolve_model_string(model_name),
        "messages": [{"role": "system", "content": system_prompt}, *st.session_state.messages],
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    if send_temperature:
        request["temperature"] = temperature
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_llm().query([request])[0]
        if not reply:
            reply = "*(empty response — check the terminal log for API errors)*"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
