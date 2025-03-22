import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community. document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

## Streamlit APP
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template=""" 
Provide a summary of thr following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url")

    else:
        try:
            with st.spinner("Waiting..."):

                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"})
                    
                docs=loader.load()

                chain=load_summarize_chain(llm, chain_type="stuff", prompt = prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")