import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Helper functions
def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def generate_deepseek_response():
    """Generate response using DeepSeek R1 model"""
    dialogue_history = "\n\n".join([msg["content"] for msg in st.session_state.messages])
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

def process_response(response):
    """Process streaming response with enhanced cleaning"""
    full_response = ""
    thinking_content = ""
    answer_content = ""
    is_thinking = False
    
    # Containers for display ordering
    answer_container = st.empty()
    think_container = st.empty()

    # Process stream chunks
    for item in response:
        text = str(item)
        full_response += text
        
        # Handle thinking tags
        if "<think>" in text:
            is_thinking = True
            text = text.replace("<think>", "")
            
        if "</think>" in text:
            is_thinking = False
            text = text.replace("</think>", "")
            thinking_content += text
            with think_container.expander("Thinking Process", expanded=False):
                st.markdown(thinking_content.strip())
            thinking_content = ""
            
        if is_thinking:
            thinking_content += text
            with think_container.expander("Thinking Process", expanded=True):
                st.markdown(thinking_content)
        else:
            # Clean from both ends and remove special chars
            clean_text = text.replace("<think>", "").replace("</think>", "").strip(' .\n\r\t')
            answer_content += clean_text + ' '

    # Final answer processing
    final_answer = answer_content.strip()
    
    # Ensure proper formatting
    if final_answer:
        # Add final punctuation if missing
        if not final_answer.endswith(('.', '!', '?')):
            final_answer += '.'
        # Collapse multiple spaces
        final_answer = ' '.join(final_answer.split())
    
    # Structure output
    if thinking_content.strip():
        return f"{final_answer}<think>{thinking_content.strip()}</think>"
    return final_answer

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    
    # API key handling
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

    # Model parameters
    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        if "<think>" in content:
            parts = content.split("<think>")
            answer_part = parts[0].strip()
            think_part = parts[1].split("</think>")[0]
            
            st.markdown(answer_part)
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_part)
        else:
            st.markdown(content.strip())

# User input handling
if prompt := st.chat_input(disabled=not replicate_api):
    clean_prompt = prompt.strip()
    
    # Validate input
    if clean_prompt and any(c.isalnum() for c in clean_prompt):
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)
    elif clean_prompt:
        st.toast("Message must contain at least one letter or number", icon="⚠️")

# Generate responses
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        response = generate_deepseek_response()
        processed_response = process_response(response)
        st.session_state.messages.append({"role": "assistant", "content": processed_response})

# Clear chat button
with st.sidebar:
    st.button(
        'Clear Chat History',
        on_click=clear_chat_history,
        type="primary" if len(st.session_state.messages) > 1 else "secondary",
        use_container_width=True
    )
