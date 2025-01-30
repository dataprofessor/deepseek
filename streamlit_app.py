import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Initialize session state for messages FIRST
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Helper functions
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.thinking_content = ""

def generate_deepseek_response():
    # Construct dialogue history
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        string_dialogue += f"{dict_message['content']}\n\n"
    
    response = replicate.stream(
        "deepseek-ai/deepseek-r1",
        input={
            "prompt": string_dialogue,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_tokens": st.session_state.max_tokens,
            "presence_penalty": st.session_state.presence_penalty,
            "frequency_penalty": st.session_state.frequency_penalty
        }
    )
    return response

def process_response(response, auto_collapse=True):
    full_response = ''
    thinking_text = ''
    answer_text = ''
    is_thinking = False
    
    response_container = st.empty()
    thinking_container = None

    for item in response:
        text = str(item)
        full_response += text
        
        if '<think>' in text:
            is_thinking = True
            text = text.replace('<think>', '')
            thinking_container = st.empty()
            
        if '</think>' in text:
            is_thinking = False
            text = text.replace('</think>', '')
            thinking_text += text
            
            with thinking_container.expander("Thinking Process", expanded=not auto_collapse):
                st.markdown(thinking_text)
            thinking_text = ''
            
        if is_thinking:
            thinking_text += text
            with thinking_container.expander("Thinking Process", expanded=True):
                st.markdown(thinking_text)
        else:
            answer_text += text
            response_container.markdown(answer_text)
    
    return full_response

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    st.write('This chatbot is created using the DeepSeek R1 LLM model.')
    
    # API key handling
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
        st.success('API key already provided!', icon='✅')
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter valid credentials!', icon='⚠️')
        else:
            st.success('Proceed to chat!', icon='👉')
    
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Model parameters
    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)
    
    # UI settings
    auto_collapse = st.toggle('Auto-collapse thinking process', value=True)
    st.button('Clear Chat History', on_click=clear_chat_history,
              type="primary" if len(st.session_state.messages) > 1 else "secondary")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        # Parse and display thinking process from history
        if '<think>' in content and '</think>' in content:
            parts = content.split('</think>')
            think_content = parts[0].replace('<think>', '')
            answer_content = parts[1]
            
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_content)
            st.markdown(answer_content)
        else:
            st.markdown(content)

# User input handling
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate assistant response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_deepseek_response()
        full_response = process_response(response, auto_collapse)
        
        # Store response with thinking tags
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
