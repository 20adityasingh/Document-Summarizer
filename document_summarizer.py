import os
import gradio as gr
import warnings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain # To overcome load_summarize_chain auto switching of chain_type from stuff to map reduce
from langchain_classic.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer, util 
from langchain_core.documents import Document  


# Ignoring all warnings
warnings.filterwarnings('ignore')

# Get the Google API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")


# File loading
def loader(filename):
  loader = PyPDFLoader(filename)
  policy_file = loader.load()
  return policy_file

# Splitting the Documents into Chunks
def split_documents(policy_file):
  txt_split = RecursiveCharacterTextSplitter(chunk_size = 512,  chunk_overlap = 50)
  chunks = txt_split.split_documents(policy_file)
  return chunks

# Global variables, which are initialized once
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # Vector Embedding
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4) # Gemini 2.5 Pro API
chat_memory = ConversationBufferMemory(memory_key='chat_history', return_message=True, output_key='answer') # Chat Memory
tokenizer = SentenceTransformer('all-MiniLM-L6-v2') # Tokenizing Prompt and Intent
intents = {
    'full': 'Give me a complete summary of the entire document.',
    'topic': 'Answer a specific question about a topic in the document.'
}
intents_embd = {k:tokenizer.encode(v, convert_to_tensor=True) for k, v in intents.items()} # Intent Embedding for Cosine Similarity Check

# Process to check whether file or database are new or not
def process_new_file(filepath, chat_state):
  if not chat_state:
    chat_state = {'db':None, 'files':[], 'memory':chat_memory}

  # Check whether file is new or not 
  filename = os.path.basename(filepath)
  if filename in chat_state['files']:
    return "File is already uploaded, ASK Your Question:", chat_state

  # Got a new file
  load_file = loader(filepath)
  chunks = split_documents(load_file)

  if chat_state['db'] is None:
    # First file! Create the database.
    db = Chroma.from_documents(chunks, embedding)
    message = "New DataBase Created."
  else:
    # DB exists! Add new documents to it.
    db = chat_state['db']
    db.add_documents(chunks)
    message = "DataBase Updated"

  chat_state['db'] = db
  chat_state['files'].append(filename)
  return message, chat_state

# Generate responses based on user's intent
def generate(prompt, chat_state):

  # Not mandatory, just to check
  if not chat_state or chat_state['db'] is None:
    return "Upload a PDF File."

  # db is given with updated Database
  db = chat_state['db']
  retriever = db.as_retriever()

  # Memory is loaded per User
  memory = chat_state['memory']

  # Prompt tokenization and Intent calculation
  prompt_embd = tokenizer.encode(prompt, convert_to_tensor=True)
  intent_query_similarity = {k:util.cos_sim(prompt_embd, intent_embd).item() for k, intent_embd in intents_embd.items()}
  intent = max(intent_query_similarity, key=intent_query_similarity.get)

  # User Intent Check
  if intent == 'full':

    # Getting data from db
    all_data = db.get()
    doc_strings = all_data['documents']
    doc_metadatas = all_data['metadatas']

    # Formatting this db output as required by summarize_chain
    chunks = [Document(page_content=text, metadata=meta) for text, meta in zip(doc_strings, doc_metadatas)]

    prompt_template = """Write a concise summary of the following text:

        "{text}"

        CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Create the chains
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    try:
        # This will make exactly ONE API call
        result = stuff_chain.invoke(chunks)
        return result['output_text']
    except Exception as e:
        # This will now catch the error if the doc is too big
        if "model's context window" in str(e).lower():
            return "Error: The document is too large to summarize all at once. Please ask about a specific topic."
        else:
            return f"An error occurred: {e}"
  
  else:
    # RAG based Q&A chain 

    t_AI = ConversationalRetrievalChain.from_llm(
      llm=llm,
      chain_type='stuff',
      retriever=retriever,
      memory=memory,
      get_chat_history = lambda h : h,
      return_source_documents=False,
      return_generated_question=True,
    )

    result = t_AI.invoke(prompt)
    return result['answer']

# Gradio
with gr.Blocks() as gb:
  chat_state = gr.State({
    'db':None,
    'files':[],
    'memory':chat_memory
  })

  with gr.Row():
    gr.Column(scale=1)

    with gr.Column(scale=4):

      gr.Markdown("# 📄 AI Document Summarizer")
        
      with gr.Row():
          upload_file = gr.File(label="Upload PDF File here.", file_count='single', file_types=['.pdf'])
          status = gr.Textbox(label="Status")
      

      query_area = gr.Textbox(label="Write Your Query Here.")
      answer = gr.Textbox(label="Your Result", lines=7)

    gr.Column(scale=1)

  upload_file.upload(
    fn=process_new_file,
    inputs=[upload_file, chat_state],
    outputs=[status, chat_state]
  )

  query_area.submit(
    fn=generate,
    inputs=[query_area, chat_state],
    outputs=[answer]
  )


gb.queue()
gb.launch(inbrowser=True)

