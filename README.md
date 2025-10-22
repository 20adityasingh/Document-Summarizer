#  AI Document Summarizer & Q&A

An intelligent PDF summarizer built with Gradio, LangChain, and Google's Gemini 2.5 Pro. This app supports multi-PDF uploads, conversation memory, and automatically detects user intent to either provide a full document summary or answer specific questions using a Retrieval-Augmented Generation (RAG) pipeline.

##  Key Features

* **Multi-PDF Upload:** Upload and process multiple PDF documents.

* **Intelligent Intent Detection:** Automatically detects whether you want a full summary or an answer to a specific question.

* **Two Modes in One:**

  * **Full Document Summarization:** Generates a concise summary of all uploaded documents combined.

  * **Conversational Q&A (RAG):** Ask specific, targeted questions about the PDF content.

* **Chat Memory:** Remembers the context of your questions (in Q&A mode) for natural follow-up.

* **Persistent Vector Store:** New documents are added to the existing Chroma vector database, allowing you to query a growing knowledge base.

##  Technology Stack

* **Frontend:** [Gradio](https://www.gradio.app/)

* **LLM:** Google Gemini 2.5 Pro (via `langchain-google-genai`)

* **Orchestration:** [LangChain](https://www.langchain.com/)

* **Embeddings:** Hugging Face `all-MiniLM-L6-v2` (via `SentenceTransformers`)

* **Vector Store:** [ChromaDB](https://www.trychroma.com/)

##  How It Works

This project has a unique feature: it intelligently routes your request based on your intent.

1. When you submit a query, the app uses a `SentenceTransformer` to calculate the cosine similarity between your query and two predefined intents:

   * **Full Summary Intent:** "Give me a complete summary of the entire document."

   * **Q&A Intent:** "Answer a specific question about a topic in the document."

2. Based on the highest similarity, it dynamically triggers one of two LangChain chains:

   * **If "Full Summary" is detected:** It loads *all* text chunks from the vector store and passes them to a `StuffDocumentsChain` to generate a single, comprehensive summary.

   * **If "Q&A" is detected:** It uses a `ConversationalRetrievalChain` (RAG) to find the most relevant document chunks, which are then used by the LLM to answer your specific question.

##  Setup & Installation

Follow these steps to run the project locally.

### 1. Clone the Repository

git clone https://github.com/your-username/your-repository-name.git cd your-repository-name
