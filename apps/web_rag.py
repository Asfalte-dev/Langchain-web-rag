import os
from xml.dom.minidom import Document
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langsmith import Client
import tempfile


st.set_page_config(
    page_title="Web Rag", 
    page_icon="🤖"
)
st.title("Web Rag")

#Load environnement variable
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPEN_API_KEY is not set in the environnement variables")
    st.stop()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

input_mode = st.radio(
    "Select input mode",
    ["Website URL", "File upload"],
    horizontal=True
)

documents = []

if input_mode == "Website URL":
    url = st.text_input("Enter a URL to load documents form :",
                    value="https://fr.wikipedia.org/wiki/Elmer,_l%27%C3%A9l%C3%A9phant_bariol%C3%A9")

    #Indexing pipeline
    if st.button("Initialize RAG system"):
        with st.spinner("Loading and processing the data..."):
            loader = WebBaseLoader(url)
            documents = loader.load()
            documents.extend(documents)
else:
    upload_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

    if st.button("Initialize RAG system from file"):
        with st.spinner("Loading and processing the document you provided..."):
            if upload_file is not None:
                file_type = upload_file.name.split('.')[-1]
                if file_type == "txt":
                    raw_text = upload_file.read().decode('utf-8')
                    documents.append(Document(page_content=raw_text))
                elif file_type == "pdf":
                    #Create a temporary file to store the uploaded pdf (because langchain pdf loader can't read in memory)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(upload_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    #Load tmp file with PyPDF
                    try: 
                        loader = PyPDFLoader(tmp_path)
                        documents.extend(loader.load())
                    #Clean tmp file path
                    finally:
                        os.unlink(tmp_path)
            else:
                st.error("Please upload a file")
                st.stop

#Embedding and store iside the vector DB
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                chunk_overlap=200, 
                                                separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") #OpenAI embedding model
    st.session_state.vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
    st.success("Rag system is initialized !")

#If indexing worked then allow to ask question
if st.session_state.vectorstore is not None:
    llm = ChatOpenAI(model="gpt-5.4")
    
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt:50442af1")

    chain = prompt | llm

    col1, col2 = st.columns(2)

    #Ask a question pipeline
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question :")

        if st.button("Get answer"):
            if question:
                with st.spinner("Generating your answer..."):
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.invoke(question)
                    docs_content = "\n\n".join(doc.page_content for doc in docs)

                    response = chain.invoke({
                        "question": question,
                        "context": docs_content
                        })
                    
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
            else:
                st.warning("Please enter a question:")

    #answer with context
    with col2:
        st.subheader("Answer")
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)

            with st.expander("Show retreived context"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**Relevant documents {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")