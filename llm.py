from langchain_community.llms import Ollama
from openai import OpenAI
from dotenv import load_dotenv
import os

# Memuat file .env
load_dotenv()
base_url = os.getenv('BASE_URL')
openai_api_key = os.getenv('OPENAI_API')


def chat_llm(question: str, model = 'gemma2'):
    
    llm = Ollama(base_url=base_url, model=model, verbose=True)
    result = llm.invoke(question)

    return result

def chat_openai(question: str, model = 'gpt-3.5-turbo-0125'):

    gpt_openai = OpenAI(api_key=openai_api_key)
    result = gpt_openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    return result.choices[0].message.content 

