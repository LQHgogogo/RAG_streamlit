from time import sleep
from rag import RagService
import streamlit as st
from streamlit.runtime.state import session_state
import config_data

st.title("智能问答")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])
prompt = st.chat_input()

if prompt:

    st.chat_message("user").write(prompt)

    st.session_state["messages"].append({"role":"user","content":prompt})

    with st.spinner("思考中"):
        response = st.session_state["rag"].chain.invoke({"input":prompt},config_data.session_config)
        st.chat_message("assistant").write(response)
        st.session_state["messages"].append({"role":"assistant","content":response})
