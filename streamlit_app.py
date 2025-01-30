import streamlit as st
import replicate
import os

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

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating DeepSeek response
def generate_deepseek_response(prompt_input):
    # Format the conversation history
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
        answer_placeholder = st.empty()
        status_container = None
        thinking_content_placeholder = None
        
        with st.spinner("Thinking..."):
            response = generate_deepseek_response(prompt)
            full_response = ''
            in_think = False
            think_buffer = ''
            answer_buffer = ''

            for item in response:
                chunk = str(item)
                full_response += chunk

                if not in_think:
                    # Check for opening <think> tag
                    think_start = full_response.find('<think>')
                    if think_start != -1:
                        # Transition to thinking phase
                        in_think = True
                        # Display content before <think> as answer
                        answer_part = full_response[:think_start]
                        if answer_part:
                            answer_placeholder.markdown(answer_part)
                        # Initialize thinking status
                        status_container = st.status("Thinking...")
                        thinking_content_placeholder = status_container.empty()
                        # Start accumulating thinking content
                        think_buffer = full_response[think_start + len('<think>'):]
                    else:
                        # Accumulate answer content
                        answer_buffer += chunk
                        answer_placeholder.markdown(answer_buffer + "▌")
                else:
                    # Inside thinking phase
                    think_buffer += chunk
                    # Check for closing </think>
                    think_end = think_buffer.find('</think>')
                    if think_end != -1:
                        # Extract thinking content and remaining answer
                        thinking_content = think_buffer[:think_end]
                        answer_part = think_buffer[think_end + len('</think>'):]
                        # Update status with final thinking content
                        thinking_content_placeholder.markdown(thinking_content)
                        status_container.update(state="complete")
                        # Show remaining answer
                        answer_buffer += answer_part
                        answer_placeholder.markdown(answer_buffer)
                        in_think = False
                    else:
                        # Update thinking content with streaming cursor
                        thinking_content_placeholder.markdown(think_buffer + "▌")

            # Handle remaining content after stream ends
            if in_think:
                # Show accumulated thinking content if </think> not found
                thinking_content_placeholder.markdown(think_buffer)
                status_container.update(state="complete")
            elif not in_think and not answer_buffer:
                # No thinking tags found at all
                answer_placeholder.markdown(full_response)

            # Store final answer
            final_answer = answer_buffer if answer_buffer else full_response
            message = {"role": "assistant", "content": final_answer}
            st.session_state.messages.append(message)
