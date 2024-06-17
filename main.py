import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from openai import OpenAI

NVIDIA_NMI_KEY = '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'

# Sidebar contents
with st.sidebar:
    st.title(':green[Quick Book Search]')
    st.markdown('''
    ## About
    Quick Book Search (QBS) is a chatbot built for querying PDF textbooks for students. It is powered by NVIDIA-NIM and LangChain.
    ''')
    add_vertical_space(5)
    st.write('Made by [ROSHAN GOPALAKRISHNAN](https://github.com/roshan-gopalakrishnan/QuickBookSearch)')



# upload a pdf file
def upload_pdf():
    pdf = st.file_uploader("Upload your book", type="pdf")
    return pdf

# convert the pdf to text chunks
def process_text(pdf, chuck_size, chuck_overlap):
    pdf_reader = PdfReader(pdf)
    # print("Number of pages for uploaded PDF document: ", len(pdf_reader.pages))
    # extract the text from the PDF
    page_text = ""
    for page in pdf_reader.pages:
        page_text += page.extract_text()
    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chuck_size,
        chunk_overlap=chuck_overlap,
        length_function=len
        )
    chunks = text_splitter.split_text(text=page_text)
    if chunks:
        return chunks
    else:
        raise ValueError("Could not process text in PDF")

# find or create the embeddings
def get_embeddings(chunks, pdf):
    store_name = pdf.name[:-4]
    # check if vector store already exists
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
        st.write("Embeddings loaded from disk")
    #else create embeddings and save to disk
    else:
        # embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # create vector store to hold the embeddings
        vector_store = FAISS.from_texts(chunks, embedding=hf)
        wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vector_store)
        # save the vector store to disk
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        st.write("Embeddings saved to disk")
    if vector_store is not None:
        return vector_store
    else:
        raise ValueError("Issue creating and saving vector store")

# retrieve the docs related to the question
def retrieve_docs(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    if len(docs) == 0:
        raise Exception("No documents found")
    else:
        return [''.join(docs[i].page_content.replace("\n", "")) for i in range(len(docs)) ]
    

# generate the response
def generate_response(docs, question):
    user_content = "Context:\n"+ "\n".join(docs) + "\n\nQuestion: " + question
    system_message = {"role": "system", "content": "In the context are documents that should contain an answer. Please always reference document id (in squere brackets, for example [0],[1]) of the document that was used to make a claim. Use as many citations and documents as it is necessary to answer question. Answer user's question using documents given in the context."}
    messages = [system_message, {"role": "user", "content": user_content}]
    
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = NVIDIA_NMI_KEY
        )
    
    completion = client.chat.completions.create(
        model = "meta/llama3-8b-instruct",
        messages = messages,
        temperature = 0.1,
        top_p = 1,
        max_tokens = 1024,
        stream = True
        )
    
    _response = []

    for chunk in completion:
        # print("Chunks: ", chunk)
        if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            response = chunk.choices[0].delta.content
            _response.append(response)
    
    # print("Response: ", "".join(_response))

    return "".join(_response)



def main():
    st.write("Hello friend!")
    st.header("Search from your uploaded book!")
    # upload a PDF file
    pdf = upload_pdf()
    # check for pdf file
    if pdf is not None:
        # process text in pdf and convert to chunks
        chuck_size = 500
        chuck_overlap = 100
        try:
            chunks = process_text(pdf, chuck_size, chuck_overlap)
            vector_store = get_embeddings(chunks, pdf)
        except Exception as e:
            st.error(e)
        # ask a question
        question = st.text_input("What do you want to ask from your book?")
        if question:
            # get the docs related to the question
            docs = retrieve_docs(question, vector_store)
            # st.write("Retrieved text: ", docs)
            response = generate_response(docs, question)
            st.subheader('Response is generated by an AI model. Please verify the response by checking the right content from the book.')
            st.write(response)



if __name__ == '__main__':
    main()
