import streamlit as st
import requests

st.title("RAG System - Q&A Chat")

st.header("Ask the Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                st.json(message["citations"])

# React to user input
if query := st.chat_input("Enter your question (Multilingual supported):"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                resp = requests.post("http://localhost:8000/ask", json={"query": query}).json()
                answer = resp.get("answer", "Error retrieving answer.")
                citations = resp.get("citations", [])
                
                st.markdown(answer)
                if citations:
                    with st.expander("Sources"):
                        st.json(citations)
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

st.divider()

st.header("Document Contradiction Check")
doc1 = st.text_input("Document 1 Name (e.g., Mclaren Racing Limited.txt)")
doc2 = st.text_input("Document 2 Name (e.g., Red Bull Racing Limited.txt)")
topic = st.text_input("Topic to compare:")
if st.button("Check for Conflict"):
    with st.spinner("Analyzing..."):
        try:
            resp = requests.post("http://localhost:8000/contradict", json={
                "doc1_name": doc1, "doc2_name": doc2, "topic": topic
            }).json()
            st.write("### Analysis")
            st.write(resp.get("analysis", "Error during analysis."))
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")