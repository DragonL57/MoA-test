import os
import json
import requests
import openai
import copy
import types
import logging
import streamlit as st
from loguru import logger
import concurrent.futures

DEBUG = int(os.environ.get("DEBUG", "0"))

def generate_together(
    model,
    messages,
    max_tokens=4096,
    temperature=0.7,
    streaming=True,
):
    endpoint = "https://api.together.xyz/v1/chat/completions"
    api_key = os.environ.get('TOGETHER_API_KEY')

    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        "stream": streaming
    }

    response = requests.post(endpoint, json=data, headers=headers, stream=streaming)
    response.raise_for_status()

    if streaming:
        return response.iter_lines()
    else:
        try:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Response content: {response.content}")
            raise

def process_stream(stream, response_container):
    response_text = ""
    for line in stream:
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line.startswith("data: "):
                try:
                    chunk = json.loads(decoded_line[6:])
                    if 'choices' in chunk and len(chunk['choices']) > 0 and 'delta' in chunk['choices'][0]:
                        if 'content' in chunk['choices'][0]['delta']:
                            response_text += chunk['choices'][0]['delta']['content']
                            response_container.markdown(response_text)
                            logger.info(f"Streaming chunk: {chunk['choices'][0]['delta']['content']}")
                except json.JSONDecodeError as e:
                    if decoded_line == "data: [DONE]":
                        break
                    logger.error(f"Streaming JSON decode error: {e}")
                    logger.error(f"Streaming line content: {decoded_line}")
            elif decoded_line == "[DONE]":
                break
    return response_text

def inject_references_to_messages(messages, references):
    messages = copy.deepcopy(messages)
    system = f"""You have been provided a set of responses from various open-source models for the latest user query. Your task is to synthesize these responses into a single high-quality answer. Critically evaluate the information provided in these responses, recognizing that some information may be biased or incorrect. Your answer should not merely copy the given responses but provide a refined, accurate, and comprehensive answer to the request. Ensure your answer is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Keep the original terminology and ensure the meaning and context are preserved.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages

def clean_response(response_text):
    return response_text.replace('\n[im_start]', '').replace('\n[im_end]', '').replace('lim_start', '').replace('lim_end', '')

def generate_initial_responses(models, messages, temperature, max_tokens, generate_fn):
    initial_responses = []

    def get_response(model):
        logger.info(f"Generating initial response using proposer model {model}")
        response = generate_fn(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False  # Ensure streaming is False for proposer models
        )
        if isinstance(response, types.GeneratorType):
            full_response = "".join(response)
        else:
            full_response = response
        cleaned_response = clean_response(full_response)
        logger.info(f"Proposer model {model} generated response: {cleaned_response[:200]}")
        return cleaned_response

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model = {executor.submit(get_response, model): model for model in models}
        for future in concurrent.futures.as_completed(future_to_model):
            try:
                initial_responses.append(future.result())
            except Exception as e:
                logger.error(f"Error generating response for model {future_to_model[future]}: {e}")

    return initial_responses

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=4096,
    temperature=0.7,
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    initial_responses = generate_initial_responses(st.session_state.selected_models, messages, temperature, max_tokens, generate_fn)
    combined_references = "\n\n".join(initial_responses)
    logger.info(f"Combined references (first 200 chars): {combined_references[:200]}")

    messages = inject_references_to_messages(messages, [combined_references])
    logger.info(f"Messages after injecting combined references: {messages}")

    try:
        response_container = st.empty()
        response = generate_fn(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(f"Generated final response (first 200 chars): {response[:200] if not isinstance(response, types.GeneratorType) else 'generator'}")

        if isinstance(response, types.GeneratorType):
            full_response = process_stream(response, response_container)
            logger.info(f"Full response (first 200 chars): {full_response[:200]}")
            return full_response
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        if e.response is not None:
            logger.error(f"Response content: {e.response.content}")
        raise
