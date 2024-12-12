import streamlit as st
import PyPDF2
import torch
from transformers import AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGComparison:
    def __init__(self):
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # QA Model without context
        self.zero_shot_model = pipeline(
            "text-generation", 
            model="gpt2"
        )
        
        # QA Model with context
        self.rag_model = pipeline(
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
    
    def generate_zero_shot_response(self, query):
        """Generate response without context"""
        response = self.zero_shot_model(
            query, 
            max_length=150, 
            num_return_sequences=1,
            do_sample=True
        )[0]['generated_text']
        return response
    
    def generate_rag_response(self, context, query):
        """Generate response with context using RAG"""
        result = self.rag_model({
            'context': context,
            'question': query
        })
        return result['answer']

def main():
    st.title("RAG vs Zero-Shot Comparison")
    
    # Sidebar for input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method", 
        ["PDF Upload", "Direct Text Input"]
    )
    
    # Instantiate the RAG application
    rag_app = RAGComparison()
    
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
            # Comparison Container
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Zero-Shot Response")
                st.warning("Response without context")
                
                # Generate Zero-Shot Response
                zero_shot_response = rag_app.generate_zero_shot_response(query)
                st.write(zero_shot_response)
            
            with col2:
                st.subheader("RAG Response")
                st.success("Response with context")
                
                # Find most relevant chunk
                relevant_chunk = rag_app.find_most_relevant_chunk(query, chunks, embeddings)
                
                # Generate RAG Response
                rag_response = rag_app.generate_rag_response(relevant_chunk, query)
                st.write(rag_response)
            
            # Show Relevant Context
            with st.expander("See Relevant Context"):
                st.write(relevant_chunk)

    # Additional information
    st.sidebar.markdown("### How to Use")
    st.sidebar.info(
        "1. Choose input method (PDF or Text)\n"
        "2. Upload PDF or paste text\n"
        "3. Ask a question about the content\n"
        "4. Compare Zero-Shot vs RAG responses"
    )

if __name__ == "__main__":
    main()