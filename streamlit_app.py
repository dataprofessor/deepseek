import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Helper functions
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.thinking_content = ""

def generate_deepseek_response():
    # Retrieve the last user message from chat history
    last_user_message = None
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            last_user_message = message["content"]
            break
    if not last_user_message:
        return "No user message found."
    
    # Construct the dialogue history
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += f"{dict_message['content']}\n\n"
        else:
            string_dialogue += f"{dict_message['content']}\n\n"
    
    response = replicate.stream(
        "deepseek-ai/deepseek-r1",
        input={
            "prompt": f"{string_dialogue}{last_user_message}",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
    )
    return response

# Processes streaming response and handles thinking/answer display
def process_response(response, auto_collapse=True):
    full_response = ''
    thinking_text = ''
    answer_text = ''
    is_thinking = False
    
    # Use a container for the assistant's entire response
    response_container = st.empty()
    
    for item in response:
        text = str(item)
        full_response += text
        
        if '<think>' in text:
            is_thinking = True
            text = text.replace('<think>', '')
            
        if '</think>' in text:
            is_thinking = False
            text = text.replace('</think>', '')
            thinking_text += text  # Add final part before closing tag
            
            # Store thinking process in session state
            st.session_state.thinking_content = thinking_text
            thinking_text = ''
            
        if is_thinking:
            thinking_text += text
        else:
            answer_text += text
        
        # Update display with both components
        with response_container.container():
            if st.session_state.thinking_content:
                with st.expander("Thinking Process", expanded=not auto_collapse):
                    st.markdown(st.session_state.thinking_content)
            st.markdown(answer_text)
    
    return full_response

# Message display logic
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Parse stored messages for think tags
        content = message["content"]
        
        if '<think>' in content and '</think>' in content:
            think_content = content.split('<think>')[1].split('</think>')[0]
            answer_content = content.split('</think>')[-1]
            
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_content)
            st.markdown(answer_content)
        else:
            st.markdown(content)

# Initialize session state for thinking content
if "thinking_content" not in st.session_state:
    st.session_state.thinking_content = ""


with st.sidebar:
    # App title
    st.title('🐳💬 DeepSeek R1 Chatbot')
    st.write('This chatbot is created using the DeepSeek R1 LLM model.')

    # Replicate Credentials
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Settings
    st.subheader('⚙️ Settings')
    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    max_tokens = st.sidebar.slider('Max Tokens', min_value=100, max_value=1000, value=800, step=100)
    presence_penalty = st.sidebar.slider('Presence Penalty', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    frequency_penalty = st.sidebar.slider('Frequency Penalty', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    
    # Add toggle for auto-collapse
    auto_collapse = st.toggle('Auto-collapse thinking process', value=True)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_deepseek_response()
        full_response = process_response(response, auto_collapse)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

# Determine button type after processing messages
has_chat_history = len(st.session_state.messages) > 1
st.sidebar.button(
    'Clear Chat History',
    on_click=clear_chat_history,
    type="primary" if has_chat_history else "secondary",
    use_container_width=True
)
