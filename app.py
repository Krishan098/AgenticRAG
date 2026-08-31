import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uuid
import streamlit as st
from rag import load_document, vector_store, graph, extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 Chat with your PDF")

# Upload a file
uploaded_file = st.file_uploader("Upload a PDF, TXT, or CSV file", type=["pdf", "txt", "csv"])

# Session state to store the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Handle document upload
if uploaded_file and not st.session_state.graph_ready:
    with st.spinner("Processing and indexing document..."):
        docs = load_document(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        _ = vector_store.add_documents(all_splits)
        st.session_state.graph_ready = True
        st.success("Document indexed and graph ready!")

# Chat interface
if st.session_state.graph_ready:
    st.markdown("### Conversation")

    # Display conversation history
    for msg in st.session_state.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(extract_text(msg.content))

    user_input = st.chat_input("Ask a question about the document:")
    if user_input:
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                for event in graph.stream(
                    {"messages": [user_message]},
                    stream_mode="values",
                    config=config,
                ):
                    output_message = event["messages"][-1]
            
            clean_text = extract_text(output_message.content)
            st.markdown(clean_text)

        ai_message = AIMessage(content=clean_text)
        st.session_state.messages.append(ai_message)
        st.rerun()
