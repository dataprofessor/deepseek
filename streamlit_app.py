import streamlit as st
import replicate
import os
import re

# App title
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

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
        response_container = st.container()
        with response_container:
            # Set up container for response
            with st.container():
                # Create a status container for thinking phase
                with st.status("Thinking...", expanded=True) as status:
                    thinking_placeholder = st.empty()
                    answer_placeholder = st.empty()

                    # Get the streamed response
                    response = generate_deepseek_response(prompt)
                    full_response = ''
                    thinking_content = ''
                    is_thinking = True

                    for item in response:
                        full_response += str(item)

                        # Check for thinking content
                        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                        if think_match and is_thinking:
                            thinking_content = think_match.group(1).strip()
                            thinking_placeholder.markdown(thinking_content)

                        # Check for answer content
                        answer_parts = full_response.split('</think>')
                        if len(answer_parts) > 1 and answer_parts[1].strip():
                            if is_thinking:
                                # Update status when transitioning to answer
                                status.update(label="Generated answer", state="complete", expanded=True)
                                is_thinking = False
                                # Clear thinking content
                                thinking_placeholder.empty()

                            answer_content = answer_parts[1].strip()
                            answer_placeholder.markdown(answer_content)

                    # Store the full response
                    message = {"role": "assistant", "content": full_response}
            
            # Store the full response
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
