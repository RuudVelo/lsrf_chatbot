# python 3.8 (3.8.16) or it doesn't work
# pip install streamlit streamlit-chat langchain python-dotenv
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
import os
from query.lsrf_query import LSRFQuery
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import os

persist_directory = os.environ.get("PERSIST_DIRECTORY")
#model_path = os.environ.get("MODEL_PATH")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))
openai_api_key = os.environ.get("OPENAI_API_KEY")

from constants import CHROMA_SETTINGS

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="🤖"
    )


def main():
    init()

    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)


    vector_store = Chroma(
            client_settings=CHROMA_SETTINGS,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Je bent een bot die veel kennis heeft over wielrennen")
        ]

    st.header("Your own ChatGPT 🤖")

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
                retriever = vector_store.as_retriever(search_kwargs={"k": target_source_chunks})
            chain = RetrievalQA.from_chain_type(
            chat,chain_type="stuff", retriever=retriever, return_source_documents=True
        )
            #retriever=vector_store.as_retriever(search_kwargs={"k": target_source_chunks}),# not including gives longer answers
           # #eturn_source_documents=False, chain_type="stuff"
        #)
            response = chain(user_input)
            #answer = res["result"], res["source_documents"]
            answer = ' '.join(response)
                #source = response["source_documents"]
                #answer, _ = lsrf_query.ask_question(user_input)
                #response = answer
            st.session_state.messages.append(
                AIMessage(content=answer))
                #AIMessage(content=response))
    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()