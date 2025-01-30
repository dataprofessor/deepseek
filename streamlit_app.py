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
    # Extract all thinking blocks
    think_blocks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
    
    # Remove thinking blocks from main content
    main_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    # Display main response
    st.markdown(main_content)
    
    # Display thinking blocks in expanders
    for idx, think_content in enumerate(think_blocks, 1):
        cleaned_think = format_reasoning_response(think_content)
        if cleaned_think:
            with st.expander(f"Thinking Process ({idx})", expanded=False):
                st.markdown(cleaned_think)

def process_stream(response_stream):
    """Process the streaming response with dynamic tag handling."""
    buffer = ""
    in_think_block = False
    thinking_content = []
    response_content = []
    
    think_expander = None
    response_placeholder = st.empty()
    think_placeholder = None

    for item in response_stream:
        buffer += str(item)
        
        while True:
            if not in_think_block:
                # Look for think tag start
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    # Capture content before think tag
                    response_content.append(buffer[:start_idx])
                    buffer = buffer[start_idx+7:]  # 7 is len("<think>")
                    in_think_block = True
                    think_expander = st.expander("Thinking...", expanded=True)
                    think_placeholder = think_expander.empty()
                else:
                    response_content.append(buffer)
                    buffer = ""
                    break
            else:
                # Look for think tag end
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    # Capture content within think tag
                    thinking_content.append(buffer[:end_idx])
                    buffer = buffer[end_idx+8:]  # 8 is len("</think>")
                    in_think_block = False
                    if think_expander:
                        think_expander.update(expanded=False)
                else:
                    thinking_content.append(buffer)
                    buffer = ""
                    break
            
            # Update displays
            current_response = "".join(response_content).strip()
            if current_response:
                response_placeholder.markdown(current_response)
                
            current_think = "".join(thinking_content).strip()
            if current_think and think_placeholder:
                think_placeholder.markdown(current_think)

    # Handle remaining buffer
    if buffer:
        if in_think_block:
            thinking_content.append(buffer)
        else:
            response_content.append(buffer)
    
    # Finalize displays
    final_response = "".join(response_content).strip()
    final_think = "".join(thinking_content).strip()
    
    # Ensure proper punctuation
    if final_response and final_response[-1] not in {'.', '!', '?'}:
        final_response += '.'
    
    return final_response, final_think

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
            
            response_content, thinking_content = process_stream(response_stream)
            
            # Format final response
            final_content = response_content
            if thinking_content:
                final_content += f"<think>{thinking_content}</think>"
            
            st.session_state.messages.append({"role": "assistant", "content": final_content})
            st.rerun()

# Clear chat button
with st.sidebar:
    st.button(
        'Clear Chat History',
        on_click=clear_chat_history,
        type="primary" if len(st.session_state.messages) > 1 else "secondary",
        use_container_width=True
    )
