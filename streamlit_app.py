import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Helper functions
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def generate_deepseek_response():
    dialogue_history = "\n\n".join([msg["content"] for msg in st.session_state.messages])
    
    response = replicate.stream(
        "deepseek-ai/deepseek-r1",
        input={
            "prompt": dialogue_history,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_tokens": st.session_state.max_tokens,
            "presence_penalty": st.session_state.presence_penalty,
            "frequency_penalty": st.session_state.frequency_penalty
        }
    )
    return response

def process_response(response):
    full_response = ""
    thinking_content = ""
    answer_content = ""
    is_thinking = False
    
    response_container = st.empty()
    thinking_container = None

    for item in response:
        text = str(item)
        full_response += text
        
        # Handle thinking process
        if "<think>" in text:
            is_thinking = True
            text = text.replace("<think>", "")
            thinking_container = st.empty()
            
        if "</think>" in text:
            is_thinking = False
            text = text.replace("</think>", "")
            thinking_content += text
            with thinking_container.expander("Thinking Process", expanded=False):
                st.markdown(thinking_content)
            thinking_content = ""
            
        if is_thinking:
            thinking_content += text
            with thinking_container.expander("Thinking Process", expanded=True):
                st.markdown(thinking_content)
        else:
            answer_content += text
            response_container.markdown(answer_content)
    
    return f"<think>{thinking_content}</think>{answer_content}" if thinking_content else answer_content

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    
    # API key handling
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Model parameters
    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)
    
    # Clear chat button (moved to bottom)
    st.button('Clear Chat History', 
             on_click=clear_chat_history,
             type="primary" if len(st.session_state.messages) > 1 else "secondary",
             use_container_width=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        if "<think>" in content and "</think>" in content:
            think_part = content.split("<think>")[1].split("</think>")[0]
            answer_part = content.split("</think>")[-1]
            
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_part)
            st.markdown(answer_part)
        else:
            st.markdown(content)

# User input handling (with validation)
if prompt := st.chat_input(disabled=not replicate_api):
    if prompt.strip():  # Prevent empty messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

# Generate response only if last message is from user
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        response = generate_deepseek_response()
        processed_response = process_response(response)
        st.session_state.messages.append({"role": "assistant", "content": processed_response})
