import streamlit as st
import replicate
import os

# Page configuration
st.set_page_config(page_title="🐳💬 DeepSeek R1 Chatbot")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def clear_chat_history():
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
    """Process streaming response with strict formatting"""
    full_response = ""
    thinking_content = ""
    answer_content = ""
    is_thinking = False
    
    answer_container = st.empty()
    think_container = st.empty()

    for item in response:
        text = str(item)
        full_response += text
        
        # Handle thinking tags
        if "<think>" in text.lower():
            is_thinking = True
            text = text.replace("<think>", "").replace("</think>", "")
            
        if "</think>" in text.lower():
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
            # Remove markdown headers and separators
            clean_text = text.replace("#", "").replace("---", "").strip()
            answer_content += clean_text + " "
            answer_container.markdown(answer_content.strip())

    # Final formatting
    final_answer = ' '.join(answer_content.strip().split())
    if final_answer and not final_answer[-1] in {'.', '!', '?'}:
        final_answer += '.'
    
    return f"{final_answer}<think>{thinking_content.strip()}</think>" if thinking_content.strip() else final_answer

# Sidebar configuration
with st.sidebar:
    st.title('🐳💬 DeepSeek R1 Chatbot')
    
    # API key and model settings
    replicate_api = st.secrets['REPLICATE_API_TOKEN'] if 'REPLICATE_API_TOKEN' in st.secrets else st.text_input('Enter Replicate API token:', type='password')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('⚙️ Model Settings')
    st.session_state.temperature = st.slider('Temperature', 0.01, 1.0, 0.1)
    st.session_state.top_p = st.slider('Top P', 0.01, 1.0, 1.0)
    st.session_state.max_tokens = st.slider('Max Tokens', 100, 1000, 800)
    st.session_state.presence_penalty = st.slider('Presence Penalty', -1.0, 1.0, 0.0)
    st.session_state.frequency_penalty = st.slider('Frequency Penalty', -1.0, 1.0, 0.0)

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        if "<think>" in content:
            parts = content.split("<think>")
            answer_part = parts[0].strip()
            think_part = parts[1].split("</think>")[0].strip()
            
            st.markdown(answer_part)
            with st.expander("Thinking Process", expanded=False):
                st.markdown(think_part)
        else:
            st.markdown(content)

# Handle input and responses
if prompt := st.chat_input(disabled=not replicate_api):
    if (clean_prompt := prompt.strip()) and any(c.isalnum() for c in clean_prompt):
        st.session_state.messages.append({"role": "user", "content": clean_prompt})

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
