import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGApplication:
    def __init__(self):
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_model = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
    
    def extract_text_from_pdf(self, uploaded_file):
        """Extract text from uploaded PDF"""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_embeddings(self, chunks):
        """Create embeddings for text chunks"""
        return self.embedding_model.encode(chunks)
    
    def find_most_relevant_chunk(self, query, chunks, embeddings):
        """Find the most relevant chunk to the query"""
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = np.dot(embeddings, query_embedding)
        most_similar_idx = np.argmax(similarities)
        return chunks[most_similar_idx]
    
    def answer_question(self, context, query):
        """Answer question based on context"""
        result = self.qa_model({
            'context': context,
            'question': query
        })
        return result['answer']

def main():
    st.title("RAG Question Answering App")
    
    # Sidebar for input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method", 
        ["PDF Upload", "Direct Text Input"]
    )
    
    # Instantiate the RAG application
    rag_app = RAGApplication()
    
    # Text or PDF input
    if input_method == "PDF Upload":
        # PDF Upload
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file is not None:
            # Extract text
            text = rag_app.extract_text_from_pdf(uploaded_file)
            st.success("PDF Uploaded and Processed!")
    else:
        # Direct Text Input
        text = st.text_area(
            "Enter your text", 
            height=250, 
            placeholder="Paste the text you want to query..."
        )
    
    # Proceed if text is available
    if 'text' in locals() and text:
        # Chunk and embed text
        chunks = rag_app.chunk_text(text)
        embeddings = rag_app.create_embeddings(chunks)
        
        # Question Input
        query = st.text_input("Ask a question about the text")
        
        if query:
            # Find most relevant chunk
            relevant_chunk = rag_app.find_most_relevant_chunk(query, chunks, embeddings)
            
            # Generate answer
            answer = rag_app.answer_question(relevant_chunk, query)
            
            # Display results
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Relevant Context:")
            st.write(relevant_chunk)

    # Additional information
    st.sidebar.markdown("### How to Use")
    st.sidebar.info(
        "1. Choose input method (PDF or Text)\n"
        "2. Upload PDF or paste text\n"
        "3. Ask a question about the content"
    )

if __name__ == "__main__":
    main()