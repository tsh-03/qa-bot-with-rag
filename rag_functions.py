# ------------------------------------------------------------------------------
# RAG Functions Module
# ------------------------------------------------------------------------------
# Author: Tirth Shah
# Description: This module contains all the RAG (Retrieval-Augmented Generation) 
#              functions used by both the Gradio app and the Jupyter notebook for
#              testing.
# ------------------------------------------------------------------------------

from typing import List, Union
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def get_llm() -> OllamaLLM:
    """
    Initialize the Ollama LLM (free local model).

    Returns
    -------
    OllamaLLM
        Configured Ollama language model instance.
    """

    llm = OllamaLLM(
        model="llama2",  # or "mistral", "codellama"
        temperature=0.5
    )
    return llm


def document_loader(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a PDF document.

    Parameters
    ----------
    file_path : str or Path
        Path to the PDF file to load.

    Returns
    -------
    List[Document]
        List of loaded document objects.
    """

    loader = PyPDFLoader(str(file_path))
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data: List[Document]) -> List[Document]:
    """
    Split documents into chunks.

    Parameters
    ----------
    data : List[Document]
        List of documents to split into chunks.

    Returns
    -------
    List[Document]
        List of document chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def hf_embedding() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings (free, no API key needed).

    Returns
    -------
    HuggingFaceEmbeddings
        Configured HuggingFace embeddings model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


def vector_database(chunks: List[Document]) -> Chroma:
    """
    Create vector database from chunks.

    Parameters
    ----------
    chunks : List[Document]
        List of document chunks to store in vector database.

    Returns
    -------
    Chroma
        Chroma vector database instance.
    """

    embedding_model = hf_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


def retriever(file) -> VectorStoreRetriever:
    """
    Create retriever - Gradio version with file object.

    Parameters
    ----------
    file : object
        Gradio file object with name attribute.

    Returns
    -------
    VectorStoreRetriever
        Vector store retriever instance.
    """

    splits = document_loader(file.name)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever


def retriever_qa(file, query: str) -> str:
    """
    Ask question - Gradio version with file object.

    Parameters
    ----------
    file : object
        Gradio file object with name attribute.
    query : str
        Question to ask about the document.

    Returns
    -------
    str
        Answer to the question based on the document.
    """
    
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response['result']
