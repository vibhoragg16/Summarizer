import sys
import os

st.write(f"Python path: {sys.executable}")
st.write(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
st.write(f"All sys paths: {sys.path}")


pip install pydantic --force-reinstall

import subprocess
import streamlit as st

st.write("Checking installed dependencies...")

installed_packages = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
st.text(installed_packages.stdout)  # Print installed packages

try:
    import pydantic
    st.success(f"Pydantic is installed (version: {pydantic.__version__})")
except ImportError:
    st.error("Pydantic is not installed")
    # Display installed packages
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    st.code(result.stdout)

from pydantic import HttpUrl, ValidationError
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

## Streamlit APP
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Function to validate URL using Pydantic
def is_valid_url(url: str) -> bool:
    try:
        HttpUrl(url)  # Pydantic URL validation
        return True
    except ValidationError:
        return False

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template=""" 
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the necessary information to get started")
    elif not is_valid_url(generic_url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting..."):

                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url])

                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run({"text": docs})

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
