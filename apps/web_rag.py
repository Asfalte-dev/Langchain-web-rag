from operator import itemgetter
import os
from xml.dom.minidom import Document
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import bs4
import tempfile
from langchain_community.vectorstores import Chroma



st.set_page_config(
    page_title="Web Rag", 
    page_icon="🤖"
)
st.title("Web Rag")

#Load environnement variable
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
langchain_tracing_v2 = os.environ.get("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.environ.get("LANGCHAIN_ENDPOINT")
langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY is not set in the environnement variables")
    st.stop()

if not langchain_api_key:
    st.error("LANGCHAIN_API_KEY is not set in the environnement variables")
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
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs=dict( #bs_kwargs are the arguments for BeautifulSoup to parse the webpage and extract relevant content
                    parse_only=bs4.SoupStrainer(
                        ["article", "main", "h1", "h2", "p"] #Only keep relevant tag to avoid noise in the vector DB
                    )
                )
            )
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
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-large",
        chunk_size=300, #Size of each chunk in tokens
        chunk_overlap=50 #Overlap between chunks to maintain context
        )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") #OpenAI embedding model
    st.session_state.vectorstore = Chroma(collection_name="web_rag_collection", 
                                                         embedding_function=embeddings,
                                                         persist_directory="../chroma_db") #Store chunks in Chroma vector DB & persist it on disk for future use
    
    ids= [ #Generate unique ids for each chunk based on source and index to avoid duplicates in the vector DB
        f"{chunk.metadata.get('source', 'unknown_source')}_{i}" 
        for i, chunk in enumerate(chunks)
    ]
    existing = st.session_state.vectorstore.get(ids=ids) #“Out of these IDs, which ones are already stored in the database?”
    existing_ids = set(existing["ids"])
    
    new_chunks = []
    new_ids = []

    for chunk, doc_id in zip(chunks, ids): #If this chunk’s ID is not already in Chroma, keep it.
        if doc_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(doc_id)

    if new_chunks: #Only add new chunks to the vector DB to avoid duplicates and save storage space
            st.session_state.vectorstore.add_documents(
            documents=new_chunks,
            ids=new_ids
        )
    else:
        st.info("This URL or document is already indexed.")

    st.success("Rag system is initialized !")

#If indexing worked then allow to ask question
if st.session_state.vectorstore is not None:
    
    col1, col2 = st.columns(2)
    #Ask a question pipeline
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question :")

        if st.button("Get answer"):
            if question:
                with st.spinner("Generating your answer..."):
                    retriever = st.session_state.vectorstore.as_retriever()
                    # multi query
                    multi_template = """You are an AI language model assistant that helps answering questions about GregTech:NewHorizons. Your task is to generate 1 - 5 different sub questions OR alternate versions of the given user question to retrieve relevant documents from a vector database.

                                        By generating multiple versions of the user question,
                                        your goal is to help the user overcome some of the limitations
                                        of distance-based similarity search.

                                        By generating sub questions, you can break down questions that refer to multiple concepts into distinct questions. This will help you get the relevant documents for constructing a final answer

                                        If multiple concepts are present in the question, you should break into sub questions, with one question for each concept

                                        Provide these alternative questions separated by newlines between XML tags. For example:

                                        <questions>
                                        - Question 1
                                        - Question 2
                                        - Question 3
                                        </questions>

                                        Original question: {question}"""
                    
                    prompt_perspectives = ChatPromptTemplate.from_template(multi_template)

                    def parse_questions(text: str): #Parse the output of the multi query prompt to extract the generated questions, remove empty lines and the XML tags
                        return [
                            line.strip().lstrip("- ").strip()
                            for line in text.splitlines()
                            if line.strip()
                            and not line.strip().startswith("<")
                            and not line.strip().endswith(">")
                        ]
                    
                    generate_queries = ( #This pipeline will generate multiple queries from the original question to retrieve more relevant documents from the vector DB and improve the final answer quality
                        prompt_perspectives 
                        | ChatOpenAI(temperature=0) 
                        | StrOutputParser() 
                        | parse_questions
                    )
                    llm = ChatOpenAI(model="gpt-5.4")

                    # RECURSIVE RAG DECOMPOSITON

                    template = """Here is the question you need to answer:

                                \n --- \n {question} \n --- \n

                                Here is any available background question + answer pairs:

                                \n --- \n {q_a_pairs} \n --- \n

                                Here is additional context relevant to the question: 

                                \n --- \n {context} \n --- \n

                                Use the above context and any background question + answer pairs to answer the question: \n {question}
                                """

                    decomposition_prompt = ChatPromptTemplate.from_template(template)

                    decompostion_chain = (
                        decomposition_prompt
                        | llm
                        | StrOutputParser()
                    )

                    # RAG FUSION EXAMPLE
                    
                    def reciprocal_rank_fusion(results: list[list[Document]], k=60) -> list[Document]:
                        """Reciprocal_rank_fusion taht takes multiple lists of ranked documents
                            and an optional parameter k used in the RRF formula"""

                        #Initialize a dictionary to store the scores for each document
                        fused_scores = {}

                        # Iterate trought each list of ranked documents
                        for docs in results:
                            # Iterate through each document in the list
                            for rank, doc in enumerate(docs):
                                #convert the doucment to a string format to use as a key
                                doc_str = dumps(doc)
                                #if the document is not yet in fused_score dictionary, add it with an initial score of 0
                                if doc_str not in fused_scores:
                                    fused_scores[doc_str] = 0
                                #Retreive the current score of the document
                                previous_score = fused_scores[doc_str]
                                #Update the score of the document using the RRF formula: 1 / (rank + k)
                                fused_scores[doc_str] += 1 / (rank + k)

                        #Sort the documents based on their fused scores in descending order to get the final reranked result
                        reranked_results = [
                            loads(doc)
                            for doc, _score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                        ]

                        return reranked_results

                    def format_docs(docs): #Format the retreived documents to have a better prompt for the final answer generation
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    def format_qa_pair(question, answer):
                        """Format Q and A pair"""
                        return f"Question: {question}\nAnswer: {answer}"

                    def recursively_answer(data:dict) -> dict:
                        original_question = data["question"]

                        #Parse_question, is executed inside generate_queries
                        questions = generate_queries.invoke({
                            "question": original_question
                        })

                        qa_pairs = []
                        retrieved_results = []

                        for subquestion in questions:
                            #Retrieve documents specifically for this subquestion
                            docs = retriever.invoke(subquestion)
                            retrieved_results.append(docs)

                            answer = decompostion_chain.invoke({
                                "question": subquestion,
                                "q_a_pairs": "\n\n---\n\n".join(qa_pairs),
                                "context": format_docs(docs),
                            })

                            qa_pairs.append(
                                format_qa_pair(subquestion, answer)
                            )

                        #Fuse documents retrieved for all subquestions
                        fused_docs = reciprocal_rank_fusion(retrieved_results)

                        return {
                            "docs": fused_docs,
                            "q_a_pairs": "\n\n---\n\n".join(qa_pairs),
                        }
                    # Full rag chain with multiple queries and final answer generation
                    template = """Answer the original question using the information below.
        
                                Original question:
                                {question}
                                
                                Subquestion answers:
                                {q_a_pairs}
                                
                                Retrieved context:
                                {context}
                                """

                    prompt= ChatPromptTemplate.from_template(template)

                    def build_answer_input(data):
                        recursive_result = data["recursive"]

                        return {
                            "question": data["question"],
                            "context": format_docs(recursive_result["docs"]),
                            "q_a_pairs": recursive_result["q_a_pairs"],
                        }
                    
                    answer_chain = (
                        RunnableLambda(
                            build_answer_input,
                            name="format_answer_inputs",
                        )
                        | prompt
                        | llm
                        | StrOutputParser()
                    )


                    recursive_chain = RunnableLambda(recursively_answer)

                    full_rag_chain = ( 
                        #Adds retieved documents to the original input dictionnary
                        RunnablePassthrough.assign(recursive=recursive_chain)
                        #Uses those same documents to produce the answer
                        | RunnablePassthrough.assign(answer=answer_chain)
                    ).with_config({"run_name": "MultiQueryRAG"})
                    
                    result = full_rag_chain.invoke({
                        "question": question,
                    })

                    response = result["answer"]
                    docs = result["recursive"]["docs"]
                    
                    st.session_state.last_response = response
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