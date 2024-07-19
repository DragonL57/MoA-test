import os
import json
import datasets
import threading
import time
import types
import requests
import logging
from langdetect import detect
from functools import partial
from loguru import logger
from utils import (
    generate_together,
    generate_with_references,
    DEBUG,
)
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from threading import Event, Thread
from dotenv import load_dotenv

load_dotenv()

class SharedValue:
    def __init__(self, initial_value=0.0):
        self.value = initial_value
        self.lock = threading.Lock()

    def set(self, new_value):
        with self.lock:
            self.value = new_value

    def get(self):
        with self.lock:
            return self.value

# Updated default reference models
default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "google/gemma-2-27b-it",
    "Qwen/Qwen1.5-72B",
    "meta-llama/Llama-3-70b-chat-hf"
]

# All available models
all_models = [
    "google/gemma-2-27b-it",
    "Qwen/Qwen1.5-110B-Chat",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mixtral-8x22B",
]

# Default system prompt
default_system_prompt = """Bạn là một trợ lý AI chuyên nghiệp với kiến thức sâu rộng. Hãy cung cấp câu trả lời:
1. Chính xác và dựa trên dữ liệu
2. Cấu trúc rõ ràng với các đoạn và tiêu đề (nếu cần)
3. Ngắn gọn nhưng đầy đủ thông tin
4. Sử dụng ví dụ cụ thể khi thích hợp
5. Tránh sử dụng ngôn ngữ kỹ thuật phức tạp, trừ khi được yêu cầu
Nếu không chắc chắn về thông tin, hãy nói rõ điều đó.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = [model for model in default_reference_models]

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "conversation_deleted" not in st.session_state:
    st.session_state.conversation_deleted = False

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "edit_gpt_index" not in st.session_state:
    st.session_state.edit_gpt_index = None

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

if "main_model" not in st.session_state:
    st.session_state.main_model =  "Qwen/Qwen2-72B-Instruct"

if "streaming_response" not in st.session_state:
    st.session_state.streaming_response = ""

# Set page configuration
st.set_page_config(page_title="MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .sidebar-content {
        padding: 1rem.
    }
    .sidebar-content .custom-gpt {
        display: flex.
        align-items: center.
        justify-content: space-between.
        padding: 0.5rem.
        border-bottom: 1px solid #ccc.
    }
    .sidebar-content .custom-gpt:last-child {
        border-bottom: none.
    }
    .remove-button {
        background-color: transparent.
        color: red.
        border: none.
        cursor: pointer.
        font-size: 16px.
    }
    .modal {
        display: none.
        position: fixed.
        z-index: 1.
        left: 0.
        top: 0.
        width: 100%.
        height: 100%.
        overflow: auto.
        background-color: rgb(0,0,0).
        background-color: rgba(0,0,0,0.4).
        padding-top: 60px.
    }
    .modal-content {
        background-color: #fefefe.
        margin: 5% auto.
        padding: 20px.
        border: 1px solid #888.
        width: 80%.
    }
    .close {
        color: #aaa.
        float: right.
        font-size: 28px.
        font-weight: bold.
    }
    .close:hover,
    .close:focus {
        color: black.
        text-decoration: none.
        cursor: pointer.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Made by Võ Mai Thế Long 👨‍🏫

Powered by Together.ai
"""

def clean_response(response_text):
    # Function to clean the response by removing unwanted tags or placeholders
    return response_text.replace('\n[im_start]', '').replace('\n[im_end]', '').replace('lim_start', '').replace('lim_end', '')

# Function to process streamed responses
def process_stream(stream, response_container):
    response_text = ""
    for line in stream:
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                chunk = json.loads(decoded_line[6:])["choices"][0]["delta"]["content"]
                response_text += chunk
                st.session_state.streaming_response = clean_response(response_text)
                response_container.markdown(st.session_state.streaming_response)
            elif decoded_line.startswith("event: done"):
                break
    return clean_response(response_text)

def process_fn(item, temperature=0.7, max_tokens=2048):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=False  # Set to False for reference models
    )
    
    # Collect the entire output
    full_output = "".join(output) if isinstance(output, types.GeneratorType) else output

    if DEBUG:
        logger.info(
            f"model {model}, instruction {item['instruction']}, output {full_output[:20]}",
        )

    st.write(f"Finished querying {model}.")

    return {"output": full_output}

def run_timer(stop_event, elapsed_time):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time.set(time.time() - start_time)
        time.sleep(0.1)

def main():
    # Display welcome message
    st.markdown(welcome_message)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Additional System Instructions")
        user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)
        if st.button("Update System Instructions"):
            st.session_state.user_system_prompt = user_prompt
            combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
            if len(st.session_state.messages) > 0:
                st.session_state.messages[0]["content"] = combined_prompt
            st.success("System instructions updated successfully!")

        st.header("Model Settings")
        with st.expander("Configuration", expanded=False):
            main_model = st.selectbox(
                "Main model (aggregator model)",
                all_models,
                index=all_models.index(st.session_state.main_model)
            )
            if main_model != st.session_state.main_model:
                st.session_state.main_model = main_model

            temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens", 1, 8192, 2048, 1)

            st.subheader("Reference Models")
            for ref_model in all_models:
                if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                    if ref_model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(ref_model)
                else:
                    if ref_model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(ref_model)

        if st.button("Start New Conversation", key="new_conversation"):
            st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]
            st.rerun()

        st.subheader("Previous Conversations")
        for idx, conv in enumerate(reversed(st.session_state.conversations)):
            cols = st.columns([0.9, 0.1])
            with cols[0]:
                if st.button(f"{len(st.session_state.conversations) - idx}. {conv['first_question'][:30]}...", key=f"conv_{idx}"):
                    st.session_state.messages = conv['messages']
                    st.rerun()
            with cols[1]:
                if st.button("❌", key=f"del_{idx}", on_click=lambda i=idx: delete_conversation(len(st.session_state.conversations) - i - 1)):
                    st.session_state.conversation_deleted = True

        if st.button("Download Chat History"):
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[1:]])  
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    if st.session_state.conversation_deleted:
        st.session_state.conversation_deleted = False
        st.experimental_rerun()

    st.markdown("Hello! I am MoA chatbot, please send me your questions below.")

    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        user_language = detect(prompt)

        if len(st.session_state.messages) == 2:
            st.session_state.conversations.append({
                "first_question": prompt,
                "messages": st.session_state.messages.copy()
            })

        try:
            logger.info(f"Main model: {st.session_state.main_model}")
            logger.info(f"Selected models: {st.session_state.selected_models}")

            data = {
                "instruction": [st.session_state.messages for _ in range(len(st.session_state.selected_models))],
                "references": [[] for _ in range(len(st.session_state.selected_models))],
                "model": st.session_state.selected_models,
            }
            eval_set = datasets.Dataset.from_dict(data)

            with st.spinner("Generating response..."):
                logger.info("Starting response generation process.")
                start_time = time.time()
                try:
                    with st.chat_message("assistant") as response_container:
                        response = generate_with_references(
                            model=st.session_state.main_model,
                            messages=st.session_state.messages,
                            generate_fn=generate_together
                        )
                    end_time = time.time()
                    logger.info(f"Response generation took {end_time - start_time} seconds.")
                    # Only add the final response to the chat history
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.streaming_response})
                    st.session_state.streaming_response = ""  # Reset streaming response
                except requests.exceptions.RequestException as e:
                    end_time = time.time()
                    logger.error(f"Generation error: {str(e)}")
                    if e.response is not None:
                        logger.error(f"Response content: {e.response.content}")
                    st.error(f"An error occurred during the generation process: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred during the generation process: {str(e)}")
            logger.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()
