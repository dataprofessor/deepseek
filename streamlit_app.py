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

def parse_think_blocks(content):
    """Parse content into main content and list of think blocks."""
    think_blocks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
    main_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return main_content, think_blocks

def display_message(message):
    """Display message with proper formatting."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_assistant_message(message["content"])
        else:
            st.markdown(message["content"])

def display_assistant_message(content):
    """Parse and display assistant message with thinking process."""
    main_content, think_blocks = parse_think_blocks(content)
    st.markdown(main_content)
    
    for think_content in think_blocks:
        cleaned_think = format_reasoning_response(think_content)
        if cleaned_think:
            with st.expander("Thinking Process", expanded=False):
                st.markdown(cleaned_think)

def process_buffer(buffer, in_think_block, current_think_block):
    """Process a text buffer and extract main/thinking content."""
    main_part = ""
    think_part = ""
    remaining_buffer = ""
    new_in_think_block = in_think_block
    new_current_think_block = current_think_block

    if not in_think_block:
        start_idx = buffer.find("<think>")
        if start_idx != -1:
            main_part = buffer[:start_idx]
            remaining_buffer = buffer[start_idx+7:]
            new_in_think_block = True
            new_current_think_block = ""
        else:
            main_part = buffer
    else:
        end_idx = buffer.find("</think>")
        if end_idx != -1:
            new_current_think_block += buffer[:end_idx]
            think_part = new_current_think_block
            remaining_buffer = buffer[end_idx+8:]
            new_in_think_block = False
            new_current_think_block = ""
        else:
            new_current_think_block += buffer
            remaining_buffer = ""

    return (main_part, think_part, remaining_buffer, 
            new_in_think_block, new_current_think_block)

def stream_response(response_stream):
    """Process the streaming response with real-time updates."""
    full_response = ""
    thinking_content = ""
    current_think_block = ""
    in_think_block = False
    
    response_placeholder = st.empty()
    think_container = None
    think_placeholder = None

    for chunk in response_stream:
        text_chunk = str(chunk)
        buffer = text_chunk

        while buffer:
            (main_part, think_part, remaining_buffer,
             new_in_think_block, new_current_think_block) = process_buffer(
                buffer, in_think_block, current_think_block
            )

            # Update main response
            if main_part:
                full_response += main_part
                response_placeholder.markdown(full_response + "▌")

            # Update thinking content
            if think_part:
                thinking_content += think_part
                if think_placeholder:
                    think_placeholder.markdown(think_part + "▌")
                if think_container:
                    think_container.empty()
                    think_container = None
                    think_placeholder = None

            # Handle think block state changes
            if new_in_think_block and not in_think_block:
                think_container = st.expander("Thinking...", expanded=True)
                think_placeholder = think_container.empty()
                if new_current_think_block:
                    think_placeholder.markdown(new_current_think_block + "▌")

            current_think_block = new_current_think_block
            in_think_block = new_in_think_block

            if in_think_block and think_placeholder and current_think_block:
                think_placeholder.markdown(current_think_block + "▌")

            buffer = remaining_buffer

    # Finalize displays
    response_placeholder.markdown(full_response.strip())
    if full_response.strip() and full_response.strip()[-1] not in {'.', '!', '?'}:
        full_response = full_response.strip() + '.'
    
    return f"{full_response}<think>{thinking_content}</think>"

def generate_response(messages, temperature, top_p, max_tokens, presence_penalty, frequency_penalty):
    """Generate response using Replicate API."""
    prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
    return replicate.stream(
        "deepseek-ai/deepseek-r1",
        input={
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
    )

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    st.write('This chatbot is created using the open-source DeepSeek-R1 model.')
    
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
            response_stream = generate_response(
                st.session_state.messages,
                st.session_state.temperature,
                st.session_state.top_p,
                st.session_state.max_tokens,
                st.session_state.presence_penalty,
                st.session_state.frequency_penalty
            )
            
            final_response = stream_response(response_stream)
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
