# Libraries 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.llms.ollama import Ollama
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Some inilializations 
docs = None
documents = None

if 'database' not in st.session_state:
    st.session_state['database'] = None

## Stremlit framework 
st.markdown("<h2 style='text-align: center;'>RAG + LLM powered chatbot 📚🔍🧠</h2>", unsafe_allow_html=True)

# history 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# backup history 
if 'backup_history' not in st.session_state:
    st.session_state.backup_history = []

# LLM selection 
LLM_model = st.sidebar.selectbox(
    "Select a LLM",
    ("llama3.1", "mistral","gemma2","llava"),
)

st.sidebar.subheader('To use RAG, please upload a pdf')

# Uploading PDF 
uploaded_pdf = st.sidebar.file_uploader('Upload your File', type='pdf')

# Processing the uploaded pdf 
if uploaded_pdf:

    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_pdf.getvalue())
        ile_name = uploaded_pdf.name

    # Pass the BytesIO object directly to PyPDFLoader
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    st.sidebar.success("'PDF loaded Sucessfully'", icon="✅")


# Breaking the uploaded pdf into chucks 
st.sidebar.subheader("Select chunk size")
chunksize = st.sidebar.slider("A value between 0 and 1000",value=500,min_value=0,max_value=1000,step=1)
st.sidebar.subheader("Select chunk overlap")
chunkoverlap = st.sidebar.slider("A value between 0 and 200",value=50,min_value=0,max_value=200,step=1)


# Creating chunks 
# st.sidebar.subheader('Create chunks from uploaded file')
# if st.sidebar.button('Creat chucks'):
#     if docs is not None and chunksize != 0 and chunkoverlap !=0:
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
#         documents = text_splitter.split_documents(docs)
#         st.sidebar.success("'Chunks created succesfully'", icon="✅")


# Creating embedding database from the chucks 
st.sidebar.subheader('Create embedded database from the pdf')
if st.sidebar.button('Create database'):
    if docs is not None and chunksize != 0 and chunkoverlap !=0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunkoverlap)
        documents = text_splitter.split_documents(docs)
        st.sidebar.success("'Chunks created succesfully'", icon="✅")
    
        st.session_state.database = FAISS.from_documents(documents,OllamaEmbeddings(model='nomic-embed-text'))
        st.sidebar.success("'Database created sucessfully'", icon="✅")
        

# Clear chat history
if st.sidebar.button('🧹 Clear Chat History'):
    st.session_state.backup_history = st.session_state.chat_history 
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!", icon="✅")

if st.sidebar.button('↩️ Undo Clear'):
    if st.session_state.backup_history:
        st.session_state.chat_history = st.session_state.backup_history.copy()
        st.sidebar.success("Chat history restored!", icon="✅")
    else:
        st.sidebar.warning("No history to restore.")

# response 
def get_response(User_question,chat_history,LLM_model,db):

    if db:

        search_docs = db.similarity_search(User_question)

        context = []
        for doc in search_docs:
            context.append(doc.page_content)

        template = """
        You are a helpful assistant who answers the user's questions effectively.
        Answer the following question based only on the provided context and the chat history.
        Think step by step before providing a detailed answer. 

        - Use relevent information from the context
        - You have access to the chat history: {chat_history}, but do not mention that you are using it.
        - Do not mention that you have context and chat history or that you are using it.


        <context>
        {context}
        </context>

        User Question: {User_question}
        """
    else:

        template = """
        You are a helpful assistant. You answer prompts answered by the user. 

        - You have access to the chat history: {chat_history}
        - Use this information to answer the user's questions effectively.
        - Do not mention that the chat history is stored or that you are using it.

        Respond only to the following user query:

        User Question: {User_question}
        """

    ## Prompt template 
    prompt = ChatPromptTemplate.from_template(template)

    ## llm 
    llm = Ollama(model=LLM_model)
    Output_Parser = StrOutputParser()
    chain = prompt|llm|Output_Parser

    if db:
        return chain.stream({'chat_history' :chat_history,'User_question':User_question,'context':context}) 
    else:
        return chain.stream({'chat_history' :chat_history,'User_question':User_question})


# Conversation history 
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message('user'):
            st.markdown(message.content)
    else:
        with st.chat_message('AI'):
            st.markdown(message.content)

# Message input 
User_question = st.chat_input("Say something")
if User_question:
    st.session_state.chat_history.append(HumanMessage(User_question))

    with st.chat_message("user"):
        st.markdown(User_question)
    
    with st.chat_message('AI'):
        ai_message = st.write_stream(get_response(User_question,st.session_state.chat_history,LLM_model,st.session_state.database))

    st.session_state.chat_history.append(AIMessage(ai_message))
    # st.chat_message("user").write(User_question)


# if User_question:
#     message = st.chat_message("assistant")
#     message.write_stream(chain.stream({'question':User_question}))

# streamlit run rag_llm.py


