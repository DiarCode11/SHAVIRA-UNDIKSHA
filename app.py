from openai import OpenAI
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
import os
from main import build_graph
from utils.llm import *

load_dotenv()

openai_api_key = os.getenv('OPENAI_API')
with pdfplumber.open("assets/datasets/Umum.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

def try_encodings(text):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            return text.encode(encoding).decode('utf-8')
        except:
            continue
    return text

text = try_encodings(text)  




with st.sidebar:
    st.markdown(
    """
    <div style="text-align: center;">
        <img src="assets/images/icon.webp" width="100">
        <h1>GANESHA VIRTUAL ASSISTANT</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("👩‍🦰 Shavira")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# Inisialisasi session state untuk menyimpan pesan
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Menampilkan pesan yang sudah ada di chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input dari user
if prompt := st.chat_input():
    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Menampilkan spinner sebagai animasi loading
    with st.spinner('Shavira sedang mengetik...'):
        response = build_graph(prompt)
        # response = chat_groq(prompt)
    
    # Menyimpan pesan assistant ke dalam session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Menampilkan pesan dari assistant
    st.chat_message("assistant").write(response)
