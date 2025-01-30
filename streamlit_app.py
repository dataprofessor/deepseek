import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Initialize session state for thinking content
if "thinking_content" not in st.session_state:
    st.session_state.thinking_content = ""

# Replicate Credentials
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    st.write('This chatbot is created using the DeepSeek R1 LLM model.')
    
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

    st.subheader('Model parameters')
    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    max_tokens = st.sidebar.slider('Max Tokens', min_value=100, max_value=1000, value=800, step=100)
    presence_penalty = st.sidebar.slider('Presence Penalty', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    frequency_penalty = st.sidebar.slider('Frequency Penalty', min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.thinking_content = ""

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating DeepSeek response
def generate_deepseek_response(prompt_input):
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += f"{dict_message['content']}\n\n"
        else:
            string_dialogue += f"{dict_message['content']}\n\n"
    
    response = replicate.stream(
        "deepseek-ai/deepseek-r1",
        input={
            "prompt": f"{string_dialogue}{prompt_input}",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
    )
    return response

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_deepseek_response(prompt)
        # Create containers
        thinking_container = st.empty()
        answer_container = st.empty()

        class StreamProcessor:
            def __init__(self):
                self.full_response = ''
                self.thinking_text = ''
                self.answer_text = ''
                self.is_thinking = False

            def process_stream(self):
                for item in response:
                    self.full_response += str(item)
                    
                    if '<think>' in item:
                        self.is_thinking = True
                        continue
                        
                    if self.is_thinking and '</think>' not in item:
                        self.thinking_text = item
                        yield self.thinking_text
                        
                    if '</think>' in item:
                        self.is_thinking = False
                        continue
                        
                    if not self.is_thinking and item.strip():
                        self.answer_text += item
                        answer_container.markdown(self.answer_text)

        processor = StreamProcessor()

        # Display thinking process in an expander
        with thinking_container.expander("Thinking Process", expanded=True):
            st.write_stream(processor.process_stream())

        # Final answer display
        if processor.answer_text:
            answer_container.markdown(processor.answer_text)

        message = {"role": "assistant", "content": processor.full_response}
        st.session_state.messages.append(message)
