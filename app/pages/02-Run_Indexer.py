import json

import streamlit as st
import urllib
import os
import time
import requests
from IPython.display import display, HTML
from collections import OrderedDict

from IPython.lib.pretty import pprint
from openai.error import OpenAIError
from langchain.docstore.document import Document

from components.sidebar import sidebar
from utils import (
    embed_docs,
    get_answer,
    get_answer_turbo,
    get_sources,
    parse_docx,
    parse_pdf,
    parse_txt,
    search_docs,
    text_to_docs,
    wrap_text_in_html,
    create_data_source,
    create_skillset,
    create_index,
    create_indexer,
    get_indexer_status
)
from credentials import (
    DATASOURCE_CONNECTION_STRING,
    AZURE_SEARCH_API_VERSION,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    COG_SERVICES_NAME,
    COG_SERVICES_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION

)
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

st.set_page_config(page_title="GPT Smart Search", page_icon="📖", layout="wide")
st.header("GPT Smart Search Engine")
##
sidebar()

st.markdown("---")
st.markdown("""Please upload your files to your Azure Storage Account and Run Indexer below.""")
st.markdown("---")

placeholder = st.empty()
btn1 = placeholder.button('Run Indexer', disabled=False, key='1')
btn2 = st.button('Refresh Progress', disabled=False, key='btn2')

if btn1:
    placeholder.button('Run Indexer', disabled=True, key='2')
    r = create_data_source()
    st.markdown("""Create Data Source :: """+str(r))
    r = create_skillset()
    st.markdown("""Create Skill Set :: """ + str(r))
    r = create_index()
    st.markdown("""Create Index :: """ + str(r))
    r = create_indexer()
    st.markdown("""Create Indexer :: """ + str(r))

    placeholder.button('Run Indexer', disabled=False, key='3')
#   st.button("Refresh", disabled=False, key='btn3')

elif btn2:
    # Optionally, get indexer status to confirm that it's running\n",
    r = get_indexer_status()
    st.markdown(str(r.status_code))
    st.markdown("""Status: """ + r.json().get('lastResult').get('status'))
    st.markdown("""Items Processed: """ + str(r.json().get('lastResult').get('itemsProcessed')))
