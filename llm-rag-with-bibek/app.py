import os
import tempfile
import faiss
import numpy as np
import chromadb
import ollama
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Available models configuration
system_prompt ='''
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.
'''

EMBEDDING_MODELS = {
"nomic-embed-text-v1":"nomic-ai/nomic-embed-text-v1",
"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
"multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
"all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
"all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
"paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
}

LLM_MODELS = {
"qwen2.5:7b": "qwen2.5:7b",
"mistral": "mistral:latest",
"neural-chat": "neural-chat:latest",
"codellama": "codellama",
"llama3.2:3b": "llama3.2:3b",
}

CROSS_ENCODERS = {
"ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
"ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
"ce-msmarco-MiniLM-L6-v3": "cross-encoder/ce-msmarco-MiniLM-L6-v3"
}


# Initialize session state
if 'documents_store' not in st.session_state:
    st.session_state.documents_store = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "llama3.2:3b"
if 'selected_embedding' not in st.session_state:
    st.session_state.selected_embedding = "nomic-embed-text-v1"
if 'selected_cross_encoder' not in st.session_state:
    st.session_state.selected_cross_encoder = "ms-marco-MiniLM-L-6-v2"

def process_document(uploaded_file: UploadedFile, chunk_size: int, chunk_overlap: int) -> list[Document]:
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection():
    """Creates and returns a FAISS index for storing and searching vectors."""
    if st.session_state.faiss_index is None:
        embedding_path = EMBEDDING_MODELS[st.session_state.selected_embedding]
        st.session_state.embedding_model = SentenceTransformer(embedding_path, trust_remote_code=True)
        dimension = st.session_state.embedding_model.get_sentence_embedding_dimension()
        st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    
    return st.session_state.faiss_index, st.session_state.embedding_model

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a FAISS index for semantic search."""
    index, embedding_model = get_vector_collection()
    
    st.session_state.documents_store = []
    if st.session_state.faiss_index is not None:
        dimension = embedding_model.get_sentence_embedding_dimension()
        st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    
    # Store documents and their embeddings
    embeddings = []
    for idx, split in enumerate(all_splits):
        doc_id = f"{file_name}_{idx}"
        st.session_state.documents_store.append({
            'id': doc_id,
            'content': split.page_content,
            'metadata': split.metadata
        })
        
        # Create embedding
        embedding = embedding_model.encode([split.page_content])[0]
        embeddings.append(embedding)
  
    embeddings_array = np.array(embeddings, dtype=np.float32)
    st.session_state.faiss_index.add(embeddings_array)
    
    st.success(f"Successfully processed {len(all_splits)} document chunks!")

def query_collection(prompt: str, n_results: int = 5):
    """Queries the FAISS index with a given prompt to retrieve relevant documents."""
    if not st.session_state.documents_store:
        st.error("No documents have been processed yet. Please upload and process a document first.")
        return None
        
    index, embedding_model = get_vector_collection()
    
    # Create query embedding
    query_embedding = embedding_model.encode([prompt])[0]
    query_embedding = np.array([query_embedding], dtype=np.float32)
    
    # Search for similar documents
    n_results = min(n_results, len(st.session_state.documents_store))
    distances, indices = index.search(query_embedding, n_results)
    
    # Retrieve the actual documents
    documents = [st.session_state.documents_store[int(idx)]['content'] 
                for idx in indices[0] 
                if int(idx) < len(st.session_state.documents_store)]
    
    return {
        "documents": documents,
        "distances": distances[0].tolist(),
        "indices": indices[0].tolist()
    }

def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    if not documents:
        raise ValueError("Document list cannot be empty")
        
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder(CROSS_ENCODERS[st.session_state.selected_cross_encoder])
    ranks = encoder_model.rank(prompt, documents, top_k=min(3, len(documents)))
    
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]] + "\n\n"
        relevant_text_ids.append(rank["corpus_id"])
    
    return relevant_text, relevant_text_ids

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model=st.session_state.selected_llm,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt ,
            },
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

# Streamlit UI
st.set_page_config(page_title="RAG Question Answer")

with st.sidebar:
    st.markdown("### **Configuration Settings**")
    
    # Model Selection
    st.markdown("#### Model Settings")
    selected_llm = st.selectbox(
        "Choose LLM Model",
        options=list(LLM_MODELS.keys()),
        index=list(LLM_MODELS.keys()).index(st.session_state.selected_llm)
    )
    st.session_state.selected_llm = selected_llm

    selected_embedding = st.selectbox(
        "Choose Embedding Model",
        options=list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_embedding)
    )
    st.session_state.selected_embedding = selected_embedding

    selected_cross_encoder = st.selectbox(
        "Choose Cross-Encoder Model",
        options=list(CROSS_ENCODERS.keys()),
        index=list(CROSS_ENCODERS.keys()).index(st.session_state.selected_cross_encoder)
    )
    st.session_state.selected_cross_encoder = selected_cross_encoder

    # Chunking Settings
    st.markdown("#### Chunking Settings")
    chunk_size = st.slider(
        "Chunk Size", 
        min_value=50, 
        max_value=5000, 
        value=100, 
        step=50,
        help="Size of text chunks to process"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap", 
        min_value=0, 
        max_value=chunk_size, 
        value=50, 
        step=10,
        help="Overlap between consecutive chunks"
    )
    
    uploaded_file = st.file_uploader(
        "**📑 Upload PDF files for QnA**", 
        type=["pdf"], 
        accept_multiple_files=False
    )

    process = st.button("⚡️ Process")
    
    if process and uploaded_file:
        normalize_uploaded_file_name = uploaded_file.name.translate(
            str.maketrans({"-": "_", ".": "_", " ": "_"})
        )
        all_splits = process_document(uploaded_file, chunk_size, chunk_overlap)
        add_to_vector_collection(all_splits, normalize_uploaded_file_name)

# Question and Answer Area
st.header("🗣️ RAG Question Answer")
st.markdown(f"""
Current Configuration:
- LLM: `{st.session_state.selected_llm}`
- Embedding: `{st.session_state.selected_embedding}`
- Cross-Encoder: `{st.session_state.selected_cross_encoder}`
""")

prompt = st.text_area("**Ask a question related to your document:**")
ask = st.button("🔥 Ask")

if ask and prompt:
    if not st.session_state.documents_store:
        st.error("Please upload and process a document first.")
    else:
        results = query_collection(prompt)
        if results:
            documents = results["documents"]
            
            if documents:
                # Rerank and get relevant text
                relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, documents)
                
                # Call LLM with relevant context
                st.write("Generating answer...")
                response = call_llm(context=relevant_text, prompt=prompt)
                st.write_stream(response)

                # Display results and relevant document information
                with st.expander("See retrieved documents"):
                    st.write(results)

                with st.expander("See most relevant document ids"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)
            else:
                st.warning("No relevant documents found for your question.")