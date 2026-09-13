import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Ask My Company Reports", page_icon="📊")
st.title("📊 Ask My Company Reports")
st.caption("Upload any company report (PDF) and ask questions — answers are grounded in the document with citations.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.current_file = None
    st.session_state.messages = []

if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
    with st.spinner("Reading and indexing the PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload", files=files)
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.session_state.session_id = data["session_id"]
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.success(f"Indexed {uploaded_file.name} — ask away!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.session_id is not None:
    question = st.chat_input("Ask a question about the uploaded document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                payload = {"session_id": st.session_state.session_id, "question": question}
                response = requests.post(f"{API_URL}/ask-pdf", json=payload)
                data = response.json()
                answer = data.get("answer") or f"⚠️ {data.get('error', 'Something went wrong.')}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF above to get started.")