import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq Client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in your .env file. Please add it to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Configure the Streamlit page
st.set_page_config(page_title="QA ChatBox", page_icon="💬", layout="centered")

st.title("💬 QA ChatBox")
st.caption("A lightweight, lightning-fast AI Assistant powered by Groq")

# Initialize chat history in session state if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your QA ChatBox. How can I help you today?"}
    ]

# Display historical chat messages from session state on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Message QA ChatBox..."):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 3. Generate response from Groq API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # We pass the entire message history so the model has conversational memory
            # Using stream=True gives it that real-time typing effect like ChatGPT
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # Parse the stream chunks as they arrive
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # Update the UI dynamically with a blinking cursor effect
                    message_placeholder.markdown(full_response + "▌")
            
            # Display final polished response without the cursor
            message_placeholder.markdown(full_response)
            
            # 4. Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")