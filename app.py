
# importing necessary modules
import os 
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

#reading the OpenAI Key

os.environ['OPENAI_API_KEY'] = apikey

#reading data

reader = PdfReader('database.pdf')

#data processing

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


#OpenAI processing
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")




# App framework

st.title('AssistBot: Assisting farmers better adapt to climate change!')
st.write("")
st.write("")
st.write("")
st.markdown("<h1 style='font-size: 32px;'>AssistBot: Please enter the queries you have!</h1>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")


prompt = st.text_input('Plug in your questions here') 

docs = docsearch.similarity_search(prompt)
chain.run(input_documents=docs, question=prompt)

if prompt: 
    docs = docsearch.similarity_search(prompt)
    output = chain.run(input_documents=docs, question=prompt)

    st.write(output) 

    