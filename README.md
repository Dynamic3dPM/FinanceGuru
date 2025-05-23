# FinanceGuru

FinanceGuru is an intelligent application designed to help you gain insights from your financial documents. It leverages the power of Large Language Models (LLMs) to understand and process unstructured data, allowing you to ask questions and get answers in natural language.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Running with Docker Compose](#running-with-docker-compose)
- [Running a Local Ollama LLM](#running-a-local-ollama-llm)
- [How FinanceGuru Works with Unstructured Data](#how-financeguru-works-with-unstructured-data)
- [Contributing](#contributing)
- [License](#license)

## Overview

FinanceGuru aims to simplify the way you interact with financial documents. Instead of manually sifting through pages of text, you can upload your documents and let FinanceGuru, powered by a local Ollama LLM, do the heavy lifting. Ask questions about your financial data, get summaries, or find specific information quickly and efficiently.

## Features

*   **Document Upload:** Supports various document formats for analysis.
*   **Natural Language Queries:** Interact with your documents using plain English.
*   **Local LLM Powered:** Utilizes Ollama to run LLMs locally, ensuring data privacy and control.
*   **Text-to-Speech (TTS):** (Optional) Get answers read out to you.
*   **User-Friendly Interface:** Easy-to-use web interface for seamless interaction.

## Architecture

FinanceGuru is a full-stack application composed of:

*   **Backend:** A Python-based API (likely FastAPI) responsible for document processing, LLM interaction, and serving data to the frontend.
    *   `app/services/document_parser.py`: Handles the extraction of text and relevant information from uploaded documents.
    *   `app/services/llm_service.py`: Manages communication with the Ollama LLM, sending processed data and queries, and receiving insights.
    *   `app/services/tts_service.py`: Provides text-to-speech functionality for responses.
*   **Frontend:** A Vue.js single-page application (SPA) providing the user interface.
*   **Database:** (Specify if any, e.g., for user accounts, document metadata)
*   **LLM:** Ollama, for running language models locally.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
*   [Node.js and npm](https://nodejs.org/) (for frontend development/contribution)
*   [Python](https://www.python.org/downloads/) (for backend development/contribution)
*   [Ollama](https://ollama.ai/)

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd FinanceGuru
    ```

### Backend Setup (Manual)

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure environment variables:**
    Create a `.env` file in the `backend` directory and add necessary configurations, especially the Ollama API endpoint.
    Example `.env` content:
    ```env
    OLLAMA_API_BASE_URL=http://localhost:11434
    # Add other configurations as needed
    ```
5.  **Run the backend server:**
    (Assuming FastAPI with Uvicorn, check your `main.py` for the exact command)
    ```bash
    uvicorn app.main:app --reload
    ```

### Frontend Setup (Manual)

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend  # Or from the root: cd frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Run the frontend development server:**
    ```bash
    npm run dev
    ```
    The application should now be accessible at `http://localhost:5173` (or the port specified by Vite).

### Running with Docker Compose

This is the recommended way to run the application for ease of setup.

1.  **Ensure Ollama is running locally and accessible.** (See [Running a Local Ollama LLM](#running-a-local-ollama-llm))
2.  **From the root of the project, build and start the services:**
    ```bash
    docker-compose up --build
    ```
    This command will build the images for the backend and frontend (if Dockerfiles are configured for both) and start the services defined in `docker-compose.yml`.

## Running a Local Ollama LLM

FinanceGuru relies on a locally running Ollama instance to power its LLM capabilities. This ensures your data remains private and under your control.

1.  **Install Ollama:**
    Follow the instructions on the [official Ollama website](https://ollama.ai/) to download and install it for your operating system.

2.  **Download an LLM model:**
    Ollama supports various models. You can pull a model using the command line. For example, to download Llama 2 (a popular open-source model):
    ```bash
    ollama pull llama2
    ```
    Other models like `mistral` or `phi` are also available. Check the [Ollama model library](https://ollama.ai/library) for more options.

3.  **Run Ollama (if not already running as a service):**
    Typically, Ollama runs as a background service after installation. If you need to run it manually or ensure it's active:
    ```bash
    ollama serve
    ```
    By default, Ollama serves its API at `http://localhost:11434`.

4.  **Configure FinanceGuru to use Ollama:**
    Ensure your backend application is configured to point to your local Ollama instance. This is usually done via an environment variable (e.g., `OLLAMA_API_BASE_URL=http://localhost:11434`) in the backend's `.env` file or Docker Compose configuration.

## How FinanceGuru Works with Unstructured Data

Unstructured data, such as text from PDFs, Word documents, or other financial reports, can be challenging to analyze programmatically. FinanceGuru addresses this using the following workflow:

1.  **Document Ingestion:** You upload your financial documents through the application's interface.
2.  **Text Extraction & Parsing (`document_parser.py`):**
    The backend's `document_parser.py` service is responsible for processing these uploaded files. It extracts raw text content and potentially performs initial structuring or cleaning (e.g., removing irrelevant artifacts, identifying sections). The specific parsing capabilities would depend on the libraries and methods implemented in this module.
3.  **LLM Interaction (`llm_service.py`):**
    *   Once the text is extracted, the `llm_service.py` takes over. When you ask a question or request a summary, this service formulates a prompt containing your query and the relevant extracted text from your document(s).
    *   This prompt is then sent to the locally running Ollama LLM.
    *   The LLM processes the information and generates a response based on its understanding of the text and your query.
4.  **Response Delivery:**
    The LLM's response is sent back to the backend, which then relays it to the frontend to be displayed to you. If TTS is enabled, the `tts_service.py` might convert the textual response into speech.

By leveraging an LLM, FinanceGuru can:

*   **Understand context:** LLMs are adept at understanding nuances and context within large blocks of text.
*   **Perform semantic search:** Find information relevant to your query even if the exact keywords are not present.
*   **Summarize content:** Generate concise summaries of lengthy documents.
*   **Answer questions:** Provide answers based on the information contained within the documents.

This approach allows for flexible and powerful interaction with your financial data, moving beyond simple keyword searches to a more intuitive, conversational analysis.

## Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` for guidelines. (You'll need to create this file if you want specific contribution guidelines).

## License

(Specify your license here, e.g., MIT License. If you don't have one yet, you might consider adding one like MIT or Apache 2.0.)

---

