import streamlit as st
from utils import set_api_key, train_model, answering
import torch

st.session_state["connected"] = False

if not st.session_state["connected"]:
    # Create the Streamlit app
    st.title("Chatbot Training App")

    # Sidebar for user input
    st.sidebar.header("Set API Keys")
    openai_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    pinecone_key = st.sidebar.text_input("Enter your Pinecone API key", type="password")
    pinecone_env = st.sidebar.text_input("Enter your Pinecone API environment")

    set_keys_button = st.sidebar.button("Set API Keys")

    if set_keys_button:
        if openai_key:
            set_api_key(openai_key, pinecone_key, pinecone_env)
            st.sidebar.success("API keys set successfully!")
        else:
            st.sidebar.warning("Please enter your OpenAI API key.")

    # Main content for training and asking questions
    if openai_key:
        st.header("Train Chatbot")
        source_urls = st.text_area("Enter source URLs (one URL per line)", "")
        store = st.radio("Select vector store", ("FAISS", "PINECONE"))
        train_button = st.button("Train Chatbot")

        if train_button:
            if source_urls:
                source_list = source_urls.split("\n")
                st.info("Training in progress...")

                try:
                    chain = train_model(source_list, store)
                    st.success("Chatbot trained successfully!")
                    st.session_state["connected"] = True
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
            else:
                st.warning("Please enter at least one source URL.")
    else:
        st.warning("Please set your OpenAI API key in the sidebar.")

else:
    st.header("Ask Chatbot")
    question = st.text_input("Enter your question", "")
    ask_button = st.button("Ask")

    if ask_button and question:
        try:
            chain = torch.load("chatbot_chain.pt")
            answer = answering(chain, question)
            st.success("Chatbot's Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred while getting the answer: {e}")

