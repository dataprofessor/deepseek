import streamlit as st
import replicate
import os
import re

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def format_reasoning_response(thinking_content):
    """Format assistant content by removing think tags."""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )

def display_message(message):
    """Display a single message in the chat interface."""
    role = message["role"]
    with st.chat_message(role):
        if role == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Display assistant message with thinking content if present."""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    
    if think_match:
        think_content = think_match.group(1)
        response_content = content.replace(f"<think>{think_content}</think>", "")
        think_content = format_reasoning_response(think_content)
        
        if think_content:
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_content)
        st.markdown(response_content.strip())
    else:
        st.markdown(content)

def generate_deepseek_response():
    """Generate response using DeepSeek R1 model"""
    dialogue_history = "\n\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
    )
    
    return replicate.stream(
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

def process_thinking_phase(response):
    """Process the thinking phase of the assistant's response."""
    thinking_content = ""
    with st.status("Thinking...", expanded=True) as status:
        think_placeholder = st.empty()
        
        for item in response:
            text = str(item)
            thinking_content += text
            
            if "</think>" in text.lower():
                status.update(label="Thinking complete!", state="complete", expanded=False)
                break
            
            think_placeholder.markdown(format_reasoning_response(thinking_content))
    
    return thinking_content

def process_response_phase(response):
    """Process the response phase of the assistant's response."""
    response_placeholder = st.empty()
    response_content = ""
    
    for item in response:
        text = str(item)
        response_content += text
        response_placeholder.markdown(response_content.strip())
    
    # Clean and punctuate response
    response_content = response_content.strip()
    if response_content and not response_content.endswith(('.', '!', '?')):
        response_content += '.'
    
    return response_content

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
        st.success('API key loaded!', icon="✅")
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter valid credentials!', icon="⚠️")
        else:
            st.success('Ready to chat!', icon="👉")
    
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)

# Display all messages
for message in st.session_state.messages:
    display_message(message)

# Handle user input
if prompt := st.chat_input(disabled=not replicate_api):
    clean_prompt = prompt.strip()
    
    if clean_prompt:
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            response_stream = generate_deepseek_response()
            
            # Process thinking phase
            thinking_content = process_thinking_phase(response_stream)
            
            # Process response phase
            response_content = process_response_phase(response_stream)
            
            # Combine and format final response
            final_response = f"{response_content}<think>{thinking_content}</think>"
            
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.rerun()

# Clear chat button
with st.sidebar:
    st.button(
        'Clear Chat History',
        on_click=clear_chat_history,
        type="primary" if len(st.session_state.messages) > 1 else "secondary",
        use_container_width=True
    )
