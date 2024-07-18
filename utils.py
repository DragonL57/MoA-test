import os
import json
import requests
import openai
import copy
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))

def generate_together(
    model,
    messages,
    max_tokens=2048,
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

# Function to process streamed responses
def process_stream(stream):
    response_container = st.empty()
    response_text = ""
    for line in stream:
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                try:
                    chunk = json.loads(decoded_line[6:])["choices"][0]["delta"]["content"]
                    response_text += chunk
                    response_container.markdown(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Streaming JSON decode error: {e}")
                    logger.error(f"Streaming line content: {decoded_line}")
                    break
            elif decoded_line.startswith("event: done"):
                break
    return response_text

def inject_references_to_messages(
    messages,
    references,
):
    messages = copy.deepcopy(messages)
    system = f"""Bạn đã được cung cấp một tập hợp các phản hồi từ các mô hình mã nguồn mở khác nhau cho truy vấn người dùng mới nhất. Nhiệm vụ của bạn là tổng hợp các phản hồi này thành một câu trả lời duy nhất, chất lượng cao. Điều quan trọng là phải đánh giá phê phán thông tin được cung cấp trong các phản hồi này, nhận ra rằng một số thông tin có thể bị thiên vị hoặc sai lầm. Câu trả lời của bạn không nên đơn thuần sao chép các câu trả lời đã cho mà nên cung cấp một câu trả lời tinh chỉnh, chính xác và toàn diện cho yêu cầu. Đảm bảo câu trả lời của bạn được cấu trúc tốt, mạch lạc và tuân theo các tiêu chuẩn cao nhất về độ chính xác và độ tin cậy. Đảm bảo giữ nguyên các thuật ngữ chuyên ngành và đảm bảo rằng ý nghĩa và ngữ cảnh ban đầu được giữ nguyên.

Câu trả lời từ các model:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def google_search(query, num_results=10):  # Increase number of search results
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')
    if not api_key or not cse_id:
        raise ValueError("Google API key or Custom Search Engine ID is missing")

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 400:
            logger.error("Bad Request: ", err)
        elif err.response.status_code == 401:
            logger.error("Unauthorized: ", err)
        elif err.response.status_code == 403:
            logger.error("Forbidden: ", err)
        raise
    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise

    return search_results

def extract_snippets(search_results):
    snippets = []
    if "items" in search_results:
        for item in search_results["items"]:
            snippets.append(item["snippet"])
    return snippets

def extract_full_texts(search_results):
    full_texts = []
    if "items" in search_results:
        for item in search_results["items"]:
            full_texts.append(item["snippet"] + "\n\n" + item["link"])
    return full_texts

def extract_keywords(text):
    # Tokenize và loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return keywords

def expand_query(conversation_history, current_query):
    # Trích xuất từ khóa từ lịch sử cuộc trò chuyện
    history_keywords = extract_keywords(conversation_history)
    
    # Trích xuất từ khóa từ câu hỏi hiện tại
    current_keywords = extract_keywords(current_query)
    
    # Kết hợp và loại bỏ trùng lặp
    all_keywords = list(set(history_keywords + current_keywords))
    
    # Tạo query mở rộng
    expanded_query = " ".join(all_keywords)
    
    return expanded_query

def generate_search_query(conversation_history, current_query):
    # Sử dụng model google/gemma-2-27b-it để tạo query tìm kiếm
    model = "google/gemma-2-8b-it"
    
    # Tạo prompt cho model
    system_prompt = """Bạn là một trợ lý AI chuyên nghiệp trong việc tạo query tìm kiếm. 
    Nhiệm vụ của bạn là phân tích lịch sử cuộc trò chuyện và câu hỏi hiện tại của người dùng, 
    sau đó tạo ra một query tìm kiếm ngắn gọn, chính xác và hiệu quả. 
    Query này sẽ được sử dụng để tìm kiếm thông tin trên web.
    Hãy đảm bảo query bao gồm các từ khóa quan trọng và bối cảnh cần thiết."""

    # Remove the language instruction from the user prompt
    user_prompt = f"""Lịch sử cuộc trò chuyện:
    {conversation_history}
    
    Câu hỏi hiện tại của người dùng:
    {current_query}
    
    Hãy tạo một query tìm kiếm ngắn gọn và hiệu quả dựa trên thông tin trên."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Gọi API để generate query
    generated_query = generate_together(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )

    return generated_query.strip()