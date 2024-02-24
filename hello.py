# Import necessary modules and classes
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Check if the directory exists
if not os.path.exists("./ipc-data.pdf"):
    print("The directory 'ipc-data' does not exist. Please check the path.")
else:
    # Check if there are any PDF files in the directory
    if not any(fname.endswith('.pdf') for fname in os.listdir("ipc-data")):
        print("There are no PDF files in the 'ipc-data' directory.")
    else:
        # Initialize a directory loader to load PDF documents from a directory
        loader = DirectoryLoader("ipc-data", glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
import PyPDF2
import textwrap

# Open the PDF file
with open("./ipc-data.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    document = reader.pages[0].extract_text()

# Initialize a text splitter to split documents into smaller chunks
chunk_size = 500
texts = textwrap.wrap(document, chunk_size)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

# Create a list of Document objects
documents = [Document(text) for text in texts]

# Create a Chroma vector database from the documents
db = Chroma.from_documents(documents,embedding,persist_directory='./persist_directory/')

# Define the directory where the Chroma vector database will be saved
persist_directory = "./chroma_db"

# Creating a Vector DB using Chroma DB and SentenceTransformerEmbeddings
# Initialize SentenceTransformerEmbeddings with a pre-trained model
embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

# Create a Chroma vector database from the text chunks
db = Chroma.from_documents(texts, embeddings, persist_directory='./persist_directory/')

# To save and load the saved vector db (if needed in the future)
# Persist the database to disk
# db.persist()
# db = Chroma(persist_directory="db", embedding_function=embeddings)

# Specify the checkpoint for the language model
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"

# Initialize the tokenizer and base model for text generation
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32
)
# Create a text generation pipeline
pipe = pipeline(
    'text2text-generation',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 512,
    do_sample = True,
    temperature = 0.3,
    top_p= 0.95
)
# Initialize a local language model pipeline
local_llm = HuggingFacePipeline(pipeline=pipe)
# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    return_source_documents=True,
)
# Prompt the user for a query
input_query = str(input("Enter your query:"))

# Execute the query using the QA chain
llm_response = qa_chain({"query": input_query})

# Print the response
print(llm_response['result'])
