def stream_response(response_stream):
    """Stream the response with proper content separation."""
    full_response = ""
    thinking_content = ""
    buffer = ""
    in_think_block = False
    
    response_placeholder = st.empty()
    think_container = None
    think_placeholder = None

    for chunk in response_stream:
        buffer += str(chunk)
        
        while True:
            if not in_think_block:
                # Look for think tag start
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    # Add content before think tag to main response
                    full_response += buffer[:start_idx]
                    buffer = buffer[start_idx+7:]  # 7 is len("<think>")
                    in_think_block = True
                    if not think_container:
                        think_container = st.expander("Thinking...", expanded=True)
                        think_placeholder = think_container.empty()
                else:
                    # Add entire buffer to main response
                    full_response += buffer
                    buffer = ""
                    break
            else:
                # Look for think tag end
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    # Add content before end tag to thinking content
                    thinking_content += buffer[:end_idx]
                    buffer = buffer[end_idx+8:]  # 8 is len("</think>")
                    in_think_block = False
                    think_container = None
                    think_placeholder = None
                else:
                    # Add entire buffer to thinking content
                    thinking_content += buffer
                    buffer = ""
                    break
            
            # Update displays
            response_placeholder.markdown(full_response + "▌")
            if think_placeholder:
                think_placeholder.markdown(thinking_content + "▌")

    # Handle remaining buffer
    if in_think_block:
        thinking_content += buffer
    else:
        full_response += buffer

    # Finalize display
    response_placeholder.markdown(full_response)
    if think_placeholder:
        think_placeholder.markdown(thinking_content)
    
    # Ensure proper punctuation
    full_response = full_response.strip()
    if full_response and full_response[-1] not in {'.', '!', '?'}:
        full_response += '.'
    
    return f"{full_response}<think>{thinking_content}</think>"
