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

def format_reasoning_response(content):
    """Clean thinking content and remove residual tags."""
    return re.sub(r'</?think>', '', content).strip()

def display_message(message):
    """Display message with proper formatting."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Parse and display assistant message with thinking process."""
    think_blocks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
    main_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    st.markdown(main_content)
    
    for think_content in think_blocks:
        cleaned_think = format_reasoning_response(think_content)
        if cleaned_think:
            with st.expander("Thinking Process", expanded=False):
                st.markdown(cleaned_think)

def stream_response(response_stream):
    """Stream the response with real-time updates."""
    full_response = ""
    thinking_content = ""
    in_think_block = False
    
    response_placeholder = st.empty()
    think_container = None

    for chunk in response_stream:
        text_chunk = str(chunk)
        full_response += text_chunk

        # Handle thinking blocks
        if "<think>" in text_chunk.lower():
            in_think_block = True
            text_chunk = text_chunk.replace("<think>", "")
            if not think_container:
                think_container = st.expander("Thinking...", expanded=True)
        
        if "</think>" in text_chunk.lower():
            in_think_block = False
            text_chunk = text_chunk.replace("</think>", "")
            if think_container:
                think_container.update(expanded=False)
                think_container = None

        # Update displays
        if in_think_block:
            thinking_content += text_chunk
            if think_container:
                with think_container:
                    st.markdown(format_reasoning_response(thinking_content))
        else:
            # Update main response in real-time
            response_placeholder.markdown(full_response.replace("<think>", "").replace("</think>", "") + "▌")

    # Finalize display without cursor
    response_placeholder.markdown(full_response.replace("<think>", "").replace("</think>", ""))
    
    # Ensure proper punctuation
    final_response = full_response.strip()
    if final_response and final_response[-1] not in {'.', '!', '?'}:
        final_response += '.'
    
    return final_response

# Sidebar configuration remains the same
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

# Display chat history
for message in st.session_state.messages:
    display_message(message)

# Handle user input
if prompt := st.chat_input(disabled=not replicate_api):
    clean_prompt = prompt.strip()
    
    if clean_prompt:
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)
        
        with st.chat_message("assistant"):
            response_stream = replicate.stream(
                "deepseek-ai/deepseek-r1",
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
            
            # Stream the response with real-time updates
            full_response = stream_response(response_stream)
            
            # Save to message history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
with st.sidebar:
    st.button(
        'Clear Chat History',
        on_click=clear_chat_history,
        type="primary" if len(st.session_state.messages) > 1 else "secondary",
        use_container_width=True
    )
