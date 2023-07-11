import os
from PIL import Image


import streamlit as st
from dotenv import load_dotenv
#from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from constants import CHROMA_SETTINGS

#embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
#model_path = os.environ.get("MODEL_PATH")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))
openai_api_key = os.environ.get("OPENAI_API_KEY")


def generate_response(openai_api_key, query_text):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

        # Create retriever interface
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0), chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        res = qa(query_text)
        answer, docs = res["result"], res["source_documents"]
        return answer, docs

# Page title
image = Image.open('lsrf.jpeg')

st.image(image) #, caption='Example Image')
st.title('🦜🔗 Live Slow Ride Fast Bot')

query_text = st.text_input('Stel hier je vraag:', placeholder = '')

result = []
documents_results = []
if query_text:
    answers, documents = generate_response(openai_api_key, query_text)
    result.append(answers)
    for doc in documents:
        tes = doc.metadata["source"] + ":" + doc.page_content
        documents_results.append(tes)

if len(result):
    st.info(answers)
    st.info(documents_results)     

# Form input and query
#result = []
#documents_results = []
#with st.form('myform', clear_on_submit=True):
#    openai_api_key = st.text_input('OpenAI API Key', type='password', )
#    submitted = st.form_submit_button('Submit')
#    if submitted and openai_api_key.startswith('sk-'):
#        with st.spinner('Calculating...'):
#            answers, documents = generate_response(openai_api_key, query_text)
#            result.append(answers)
#            for doc in documents:
#                 tes = doc.metadata["source"] + ":" + doc.page_content
#            
#                 documents_results.append(tes)
#            del openai_api_key

#if len(result):
#    st.info(answers)
#    st.info(documents_results)


#image = Image.open('lsrf.jpeg')

#st.image(image, caption='Example Image')

#st.write("Live Slow Ride Fast Bot")
#col1, mid, col2 = st.beta_columns([1,1,20])
#with col1:
#    st.image('lsrf.jpeg', width=60)
#with col2:
#    st.write('Ask a question to LRSF')