import streamlit as st
from streamlit_chat import message
from PIL import Image
from query.lsrf_query import LSRFQuery

def initialize_page():
    st.set_page_config(page_title='LSRF', page_icon=':books:')
    st.image(logo_image, width=80)
    st.header("LSRF Podcast Bot")
    #st.markdown("[Github](https://github.com/xxxxxx)")

with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', )
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            del openai_api_key




def handle_query_form():
    with st.form(key='query_form'):
        user_query = st.text_input('Typ hier je vraag: ', '', key='input',
                                   help='Stel hier je vraag aan de LSRF bot')
        submit_button = st.form_submit_button('LSRF bot go!')
        # Check if Enter key is pressed and run the query
    #if st.session_state.last_query != query:
    #    st.session_state.last_query = query
    return user_query, submit_button

#Form input and query


def display_chat_history():
    for i, (user_msg, ai_msg) in enumerate(zip(st.session_state['past'][::-1],
                                               st.session_state['generated'][::-1])):
        message(user_msg, is_user=True, key=f"user_{i}", avatar_style="adventurer", seed="Abby")
        message(ai_msg, key=f"ai_{i}", avatar_style="bottts", seed="Felix")

def query(question: str) -> str:
    """
    Query the LSRFQuery model with the provided question
    :param question: The question to ask the model
    :return: The answer from the model
    """
    lsrf_query = LSRFQuery()
    answer, source_docs = lsrf_query.ask_question(question)
    return answer, source_docs


logo_image = Image.open('./lsrf.jpeg')

# Initialize page and session state

st.session_state.setdefault('generated', [])
st.session_state.setdefault('past', [])

initialize_page()
user_query, submit_button = handle_query_form()

if submit_button and user_query:
    documents_results = []
    model_response, documents = query(user_query)
    st.session_state.past.append(user_query)
    st.session_state.generated.append(model_response)
    for doc in documents:
        tes = doc.metadata["source"].split('/')[-1] + ":" + doc.page_content
        documents_results.append(tes)



if submit_button and user_query:
    with st.sidebar:
        st.sidebar.header('Bron van de informatie')
        st.info("\n".join(documents_results))

# Add a disclaimer at the bottom of the main content area
st.write("<p style='font-size: 12px;'>Disclaimer: Dit is geen door LSRF ondersteunde app</p>", unsafe_allow_html=True)

display_chat_history()