from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate 
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os

persist_directory = os.environ.get("PERSIST_DIRECTORY")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))
openai_api_key = os.environ.get("OPENAI_API_KEY")

from constants import CHROMA_SETTINGS


class LSRFQuery:
    def __init__(self):
        load_dotenv()
        self.chain = self.make_chain()
        self.chat_history = []

    def make_chain(self):
        model = ChatOpenAI(
            client=None,
            model="gpt-3.5-turbo",
            temperature=0,
        )

        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

        vector_store = Chroma(
            client_settings=CHROMA_SETTINGS,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )

        general_system_template = r""" 
        Jij bent een AI assistent om vragen te beantwoorden over de Live Slow Ride Fast podcasts. Je bent een kenner op 
        wielren training. Als je het antwoord niet weet geef dan als antwoord: 
        Op basis van de gegeven informatie kan ik geen antwoord geven. Probeer nog eens op een andere manier om de vraag te 
        stellen. Het kan zijn dat ik dan wel antwoord kan geven.
        ----
        {context}
        ----
        """
 
        general_user_template = "```{question}```"
        messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        return ConversationalRetrievalChain.from_llm(
            model,
            retriever=vector_store.as_retriever(search_kwargs={"k": target_source_chunks}),# not including gives longer answers
            return_source_documents=True, combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

    def ask_question(self, question: str):
        response = self.chain({"question": question, "chat_history": self.chat_history})

        answer = response["answer"]
        source = response["source_documents"]
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, source