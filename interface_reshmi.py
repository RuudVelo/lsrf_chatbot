import streamlit as st
import pandas as pd
import json

from agent import query_agent, create_agent
from langchain.schema import AIMessage


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object."""
    return json.loads(response)


def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)


st.title("👨‍💻 Marketing Intelligence for Essent")


filepath = "reviews.csv"


query = st.text_area("Enter your question about Essent customer reviews")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = create_agent(filepath)

    # Query the agent.
    with st.spinner("Thinking..."):
        response = query_agent(agent=agent, query=query)

        # Decode the response.
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app.
        write_response(decoded_response)
        st.session_state.messages.append(AIMessage(content=response.content))
