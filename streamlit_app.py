import streamlit as st
import replicate
import os
import re

# Constants & Configuration
DEFAULT_ASSISTANT_PROMPT = "How may I assist you today?"
MODEL_ID = "deepseek-ai/deepseek-r1"
MODEL_PARAMS = {
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens": 800,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

# Session State Management
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_ASSISTANT_PROMPT}]
    
    for param, default in MODEL_PARAMS.items():
        if param not in st.session_state:
            st.session_state[param] = default

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_ASSISTANT_PROMPT}]

# Response Processing
def format_think_blocks(content):
    return re.sub(r'</?think>', '', content).strip()

def process_stream(response_stream):
    full_response = ""
    thinking_content = ""
    current_think_block = ""
    in_think_block = False
    
    response_placeholder = st.empty()
    think_container = None
    think_placeholder = None

    for chunk in response_stream:
        buffer = str(chunk)
        while buffer:
            if not in_think_block:
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    full_response += buffer[:start_idx]
                    response_placeholder.markdown(full_response + "▌")
                    think_container = st.expander("Thinking...", expanded=True)
                    think_placeholder = think_container.empty()
                    buffer = buffer[start_idx+7:]
                    in_think_block = True
                else:
                    full_response += buffer
                    response_placeholder.markdown(full_response + "▌")
                    buffer = ""
            else:
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    current_think_block += buffer[:end_idx]
                    thinking_content += current_think_block
                    if think_placeholder:
                        think_placeholder.markdown(current_think_block + "▌")
                    buffer = buffer[end_idx+8:]
                    in_think_block = False
                    current_think_block = ""
                    if think_container:
                        think_container.empty()
                else:
                    current_think_block += buffer
                    if think_placeholder:
                        think_placeholder.markdown(current_think_block + "▌")
                    buffer = ""

    response_placeholder.markdown(full_response.strip())
    return f"{full_response}<think>{thinking_content}</think>"

# Page Setup
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")
init_session_state()

# Sidebar UI
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    st.write('This chatbot is created using the open-source DeepSeek-R1 model.')
    
    # API Key Handling
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
        st.success('API key loaded!', icon="✅")
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if replicate_api:
            if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
                st.warning('Please enter valid credentials!', icon="⚠️")
            else:
                st.success('Ready to chat!', icon="👉")
    
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Model Settings
    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)

    # Clear Chat Button
    st.button(
        'Clear Chat History',
        on_click=clear_chat_history,
        type="primary" if len(st.session_state.messages) > 1 else "secondary",
        use_container_width=True
    )

# Chat Messages Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            main_content = re.sub(r'<think>.*?</think>', '', message["content"], flags=re.DOTALL).strip()
            st.markdown(main_content)
            
            think_blocks = re.findall(r'<think>(.*?)</think>', message["content"], re.DOTALL)
            for think_content in think_blocks:
                cleaned_think = format_think_blocks(think_content)
                if cleaned_think:
                    with st.expander("Thinking Process", expanded=False):
                        st.markdown(cleaned_think)
        else:
            st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input(disabled=not os.environ.get('REPLICATE_API_TOKEN')):
    clean_prompt = prompt.strip()
    if clean_prompt:
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)
        
        with st.chat_message("assistant"):
            response_stream = replicate.stream(
                MODEL_ID,
                input={
                    "prompt": "\n\n".join(
                        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
                    ),
                    "temperature": st.session_state.temperature,
                    "top_p": st.session_state.top_p,
                    "max_tokens": st.session_state.max_tokens,
                    "presence_penalty": st.session_state.presence_penalty,
                    "frequency_penalty": st.session_state.frequency_penalty
                }
            )
            
            final_response = process_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.rerun()
