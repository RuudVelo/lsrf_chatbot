# Chat with LRSF podcasts

Chat with LSRF podcasts using ChatGPT

## Installation

1. Install poetry

`pip install poetry`

2. Clone the repo

`git clone {insert github repo url}`

3. Install project dependencies

`poetry install`

4. Make a `.env` file and add the following to the file:

`OPENAI_API_KEY={YOUR_KEY}`
`PERSIST_DIRECTORY=db`
`TARGET_SOURCE_CHUNKS=4`

5. Ingest the docs you want to 'chat' with

By default this repo uses a `source_documents` folder to store the documents to be ingested. Originally I used transcripts of the LRSF podcasts. Whisper was used for this. For each episode I had 1 file (.txt)

Supported document extensions include:

- `.csv`: CSV,
- `.docx`: Word Document,
- `.doc`: Word Document,
- `.eml`: Email,
- `.epub`: EPub,
- `.html`: HTML File,
- `.md`: Markdown,
- `.pdf`: Portable Document Format (PDF),
- `.pptx` : PowerPoint Document,
- `.txt`: Text file (UTF-8),

Then run this script to ingest

```shell
python ingest.py

```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run question_answer_docs.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

6. Chat with the documents

There is a chat app using streamlit. The app can be launched with:

First, load the command line:

```shell
streamlit run app2.py
```

Note: Depending on the memory of your computer, prompt request, and number of chunks returned from the source docs, it may take anywhere from 40 to 300 seconds for the model to respond to your prompt.

You cannot use this chatbot without internet connection.

## Credits

Credit to imartinez for the privateGPT ingest logic and docs guidance [here](https://github.com/imartinez/privateGPT/blob/main/README.md?plain=1)
Credit to mayoear for private chatbot example [here](https://github.com/mayooear/private-chatbot-mpt30b-langchain)