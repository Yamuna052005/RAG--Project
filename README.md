# Chat with PDF – RAG Based Document Intelligence System

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to interact with PDF documents through natural language questions. The system extracts text from uploaded PDFs, converts it into embeddings, retrieves the most relevant context, and generates precise answers using a large language model.

## Problem It Solves

Manually searching through lengthy documents is slow and inefficient. This system enables quick information retrieval and contextual question answering directly from document content.

## Core Features

* Upload and process multiple PDF files
* Automatic text extraction from documents
* Intelligent chunking of content
* Semantic search using vector embeddings
* Context-based question answering
* Structured, concise responses in bullet points

## Technology Stack

### Frontend

* Streamlit

### LLM / AI

* Google Gemini (Generative AI)
* LangChain

### Embeddings

* Sentence Transformers (all-MiniLM-L6-v2)

### Vector Database

* FAISS

### Document Processing

* PyPDF

### Environment Management

* Python Dotenv

## System Workflow

1. User uploads one or more PDF documents.
2. Text is extracted from each page.
3. Content is split into manageable chunks.
4. Chunks are converted into embeddings.
5. FAISS stores and retrieves relevant context.
6. The LLM answers user questions using only retrieved information.

## Example Question

"Answer questions using the uploaded PDF context."

## Installation

### Clone repository
git clone <your-repository-url>

### Move into project folder
cd <project-folder>

### Create virtual environment
python -m venv venv

### Activate environment (Windows)
venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run the Application
streamlit run app.py

## Environment Variables

Create a `.env` file in the root directory and add:

GOOGLE_API_KEY=your_api_key_here
