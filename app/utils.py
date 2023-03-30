from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List

import docx2txt
import streamlit as st
from requests import Response

from embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError
from prompts import STUFF_PROMPT, REFINE_PROMPT, REFINE_QUESTION_PROMPT
from pypdf import PdfReader
import os
import json
import requests
from pprint import pprint
from credentials import (
    DATASOURCE_CONNECTION_STRING,
    AZURE_SEARCH_API_VERSION,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    COG_SERVICES_NAME,
    COG_SERVICES_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY
)


# Define the names for the data source, skillset, index and indexer
def set_indexer_params():
    datasource_name = "cogsrch-datasource"
    skillset_name = "cogsrch-skillset"
    index_name = "cogsrch-index"
    indexer_name = "cogsrch-indexer"
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}


# Create a data source
# You should already have a blob container that contains the sample data, see app/credentials.py
def create_data_source() -> int:
    datasource_name = "cogsrch-datasource"
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}
    datasource_payload = {
        "name": datasource_name,
        "description": "Demo files to demonstrate cognitive search capabilities.",
        "type": "azureblob",
        "credentials": {
            "connectionString": DATASOURCE_CONNECTION_STRING
        },
        "container": {
            "name": "documents"
        }
    }
    r = requests.put(AZURE_SEARCH_ENDPOINT + "/datasources/" + datasource_name,
                     data=json.dumps(datasource_payload), headers=headers, params=params)

    return r.status_code


# Create a skillset
def create_skillset() -> int:
    skillset_name = "cogsrch-skillset"
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}
    skillset_payload = {
        "name": skillset_name,
        "description": "Extract entities, detect language and extract key-phrases",
        "skills":
            [
                {
                    "@odata.type": "#Microsoft.Skills.Vision.OcrSkill",
                    "description": "Extract text (plain and structured) from image.",
                    "context": "/document/normalized_images/*",
                    "defaultLanguageCode": "en",
                    "detectOrientation": True,
                    "inputs": [
                        {
                            "name": "image",
                            "source": "/document/normalized_images/*"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "text",
                            "targetName": "images_text"
                        }
                    ]
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.MergeSkill",
                    "description": "Create merged_text, which includes all the textual representation of each image inserted at the right location in the content field. This is useful for PDF and other file formats that supported embedded images.",
                    "context": "/document",
                    "insertPreTag": " ",
                    "insertPostTag": " ",
                    "inputs": [
                        {
                            "name": "text", "source": "/document/content"
                        },
                        {
                            "name": "itemsToInsert", "source": "/document/normalized_images/*/images_text"
                        },
                        {
                            "name": "offsets", "source": "/document/normalized_images/*/contentOffset"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "mergedText",
                            "targetName": "merged_text"
                        }
                    ]
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                    "context": "/document",
                    "textSplitMode": "pages",
                    "maximumPageLength": 2000,  # 5000 is default
                    "inputs": [
                        {
                            "name": "text",
                            "source": "/document/content"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "textItems",
                            "targetName": "pages"
                        }
                    ]
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.LanguageDetectionSkill",
                    "description": "If you have multilingual content, adding a language code is useful for filtering",
                    "context": "/document",
                    "inputs": [
                        {
                            "name": "text",
                            "source": "/document/pages/*"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "languageName",
                            "targetName": "language"
                        }
                    ]
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
                    "context": "/document/pages/*",
                    "maxKeyPhraseCount": 2,
                    "inputs": [
                        {
                            "name": "text",
                            "source": "/document/pages/*"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "keyPhrases",
                            "targetName": "keyPhrases"
                        }
                    ]
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.V3.EntityRecognitionSkill",
                    "context": "/document",
                    "categories": ["Person", "Location", "Organization", "DateTime", "URL", "Email"],
                    "minimumPrecision": 0.3,
                    "inputs": [
                        {
                            "name": "text",
                            "source": "/document/pages/*"
                        }
                    ],
                    "outputs": [
                        {
                            "name": "persons",
                            "targetName": "persons"
                        },
                        {
                            "name": "locations",
                            "targetName": "locations"
                        },
                        {
                            "name": "organizations",
                            "targetName": "organizations"
                        },
                        {
                            "name": "dateTimes",
                            "targetName": "dateTimes"
                        },
                        {
                            "name": "urls",
                            "targetName": "urls"
                        },
                        {
                            "name": "emails",
                            "targetName": "emails"
                        }
                    ]
                }
            ],
        "cognitiveServices": {
            "@odata.type": "#Microsoft.Azure.Search.CognitiveServicesByKey",
            "description": COG_SERVICES_NAME,
            "key": COG_SERVICES_KEY
        }
    }

    r = requests.put(AZURE_SEARCH_ENDPOINT + "/skillsets/" + skillset_name,
                     data=json.dumps(skillset_payload), headers=headers, params=params)
    return r.status_code


def create_index() -> int:
    # Create an index
    # Queries operate over the searchable fields and filterable fields in the index
    datasource_name = "cogsrch-datasource"
    skillset_name = "cogsrch-skillset"
    index_name = "cogsrch-index"
    indexer_name = "cogsrch-indexer"
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}
    index_payload = {
        "name": index_name,
        "fields": [
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            },
            {
                "name": "pages",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            },
            {
                "name": "images_text",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "false"
            },
            {
                "name": "language",
                "type": "Edm.String",
                "searchable": "true",
                "sortable": "true",
                "filterable": "true",
                "facetable": "false"
            },
            {
                "name": "keyPhrases",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "persons",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "locations",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "organizations",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "dateTimes",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "urls",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            },
            {
                "name": "emails",
                "type": "Collection(Edm.String)",
                "searchable": "true",
                "sortable": "false",
                "filterable": "true",
                "facetable": "true"
            },
            {
                "name": "metadata_storage_name",
                "type": "Edm.String",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            },
            {
                "name": "metadata_storage_path",
                "type": "Edm.String",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            },
            {
                "name": "id",
                "type": "Edm.String",
                "key": "true",
                "searchable": "true",
                "sortable": "false",
                "filterable": "false",
                "facetable": "false"
            }
        ],
        "semantic": {
            "configurations": [
                {
                    "name": "my-semantic-config",
                    "prioritizedFields": {
                        "prioritizedContentFields": [
                            {
                                "fieldName": "content"
                            }
                        ]
                    }
                }
            ]
        }
    }

    r = requests.put(AZURE_SEARCH_ENDPOINT + "/indexes/" + index_name,
                     data=json.dumps(index_payload), headers=headers, params=params)
    return r.status_code


def create_indexer() -> int:
    datasource_name = "cogsrch-datasource"
    skillset_name = "cogsrch-skillset"
    index_name = "cogsrch-index"
    indexer_name = "cogsrch-indexer"
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}
    # Create an indexer
    indexer_payload = {
        "name": indexer_name,
        "dataSourceName": datasource_name,
        "targetIndexName": index_name,
        "skillsetName": skillset_name,
        "schedule": {"interval": "PT2H"},
        "fieldMappings": [
            {
                "sourceFieldName": "metadata_storage_path",
                "targetFieldName": "id",
                "mappingFunction": {"name": "base64Encode"}
            },
            {
                "sourceFieldName": "metadata_storage_path",
                "targetFieldName": "metadata_storage_path"
            },
            {
                "sourceFieldName": "metadata_storage_name",
                "targetFieldName": "metadata_storage_name"
            }
        ],
        "outputFieldMappings":
            [
                {
                    "sourceFieldName": "/document/content",
                    "targetFieldName": "content"
                },
                {
                    "sourceFieldName": "/document/pages/*",
                    "targetFieldName": "pages"
                },
                {
                    "sourceFieldName": "/document/normalized_images/*/images_text",
                    "targetFieldName": "images_text"
                },
                {
                    "sourceFieldName": "/document/language",
                    "targetFieldName": "language"
                },
                {
                    "sourceFieldName": "/document/pages/*/keyPhrases/*",
                    "targetFieldName": "keyPhrases"
                },
                {
                    "sourceFieldName": "/document/persons",
                    "targetFieldName": "persons"
                },
                {
                    "sourceFieldName": "/document/locations",
                    "targetFieldName": "locations"
                },
                {
                    "sourceFieldName": "/document/organizations",
                    "targetFieldName": "organizations"
                },
                {
                    "sourceFieldName": "/document/dateTimes",
                    "targetFieldName": "dateTimes"
                },
                {
                    "sourceFieldName": "/document/urls",
                    "targetFieldName": "urls"
                },
                {
                    "sourceFieldName": "/document/emails",
                    "targetFieldName": "emails"
                }
            ],
        "parameters":
            {
                "maxFailedItems": -1,
                "maxFailedItemsPerBatch": -1,
                "configuration":
                    {
                        "dataToExtract": "contentAndMetadata",
                        "imageAction": "generateNormalizedImages"
                    }
            }
    }
    print(DATASOURCE_CONNECTION_STRING)
    r = requests.put(AZURE_SEARCH_ENDPOINT + "/indexers/" + indexer_name,
                     data=json.dumps(indexer_payload), headers=headers, params=params)
    return r.status_code


def get_indexer_status() -> Response:
    indexer_name = "cogsrch-indexer"
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json', 'api-key': AZURE_SEARCH_KEY}
    params = {'api-version': AZURE_SEARCH_API_VERSION}
    # Optionally, get indexer status to confirm that it's running
    r = requests.get(AZURE_SEARCH_ENDPOINT + "/indexers/" + indexer_name +
                     "/status", headers=headers, params=params)
    # pprint(json.dumps(r.json(), indent=1))
    return r
    # return str(int(r.status_code)) + "\n" + "Status: " + r.json().get('lastResult').get(
    #   'status') + "\n" + "Items Processed: " + r.json().get('lastResult').get('itemsProcessed')


# @st.cache_data
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


# @st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


# @st.cache_data
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


# @st.cache_data
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# @st.cache_data(show_spinner=False)
def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("AZURE_OPENAI_API_KEY"):
        raise AuthenticationError(
            "You need to set the env variable AZURE_OPENAI_API_KEY"
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings()
        index = FAISS.from_documents(docs, embeddings)

        return index


# @st.cache_data
def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=2)
    return docs


# @st.cache_data
def get_answer(docs: List[Document],
               query: str,
               deployment: str,
               chain_type: str,
               temperature: float,
               max_tokens: int
               ) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    if deployment == "gpt-35-turbo":
        llm = AzureChatOpenAI(deployment_name=deployment, temperature=temperature, max_tokens=max_tokens)
    else:
        llm = AzureOpenAI(deployment_name=deployment, temperature=temperature, max_tokens=max_tokens)

    chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

    answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    # answer = chain( {"input_documents": docs, "question": query, "language": language}, return_only_outputs=True)

    return answer


# @st.cache_data
def get_answer_turbo(docs: List[Document],
                     query: str,
                     deployment: str,
                     language: str,
                     chain_type: str,
                     temperature: float,
                     max_tokens: int
                     ) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""
    # In Azure OpenAI create a deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)"

    # Get the answer
    if deployment == "gpt-35-turbo":
        llm = AzureChatOpenAI(deployment_name=deployment, temperature=temperature, max_tokens=max_tokens)
    else:
        llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-3.5-turbo-0301", temperature=temperature,
                              max_tokens=max_tokens)

    if chain_type == "refine":
        chain = load_qa_chain(llm, chain_type=chain_type, question_prompt=REFINE_QUESTION_PROMPT,
                              refine_prompt=REFINE_PROMPT)
        answer = chain({"input_documents": docs, "question": query, "language": language}, return_only_outputs=True)

        # passing answer again to openai to remove any additional leftover wording from chatgpt
        answer = chain({"input_documents": [Document(page_content=answer['output_text'])], "question": query,
                        "language": "English"}, return_only_outputs=False)

    if chain_type == "stuff":
        chain = load_qa_chain(llm, chain_type=chain_type, prompt=STUFF_PROMPT)

    answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    return answer


# @st.cache_data
def get_answer_turbo(docs: List[Document],
                     query: str,
                     language: str,
                     chain_type: str,
                     temperature: float,
                     max_tokens: int
                     ) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    # In Azure OpenAI create a deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)"
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-3.5-turbo-0301", temperature=temperature,
                          max_tokens=max_tokens)

    if chain_type == "refine":
        chain = load_qa_chain(llm, chain_type=chain_type, question_prompt=REFINE_QUESTION_PROMPT,
                              refine_prompt=REFINE_PROMPT)
        answer = chain({"input_documents": docs, "question": query, "language": language}, return_only_outputs=True)

        # passing answer again to openai to remove any additional leftover wording from chatgpt
        answer = chain({"input_documents": [Document(page_content=answer['output_text'])], "question": query,
                        "language": "English"}, return_only_outputs=False)

    if chain_type == "stuff":
        chain = load_qa_chain(llm, chain_type=chain_type, prompt=STUFF_PROMPT)
        answer = chain({"input_documents": docs, "question": query, "language": language}, return_only_outputs=False)

    return answer


# @st.cache_data
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
