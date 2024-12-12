import streamlit as st
import torch
from transformers import pipeline
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
    
    # Sample Story
    sample_story = """
    In the small town of Riverdale, Emma Johnson was a dedicated local veterinarian who ran a unique animal rescue center. One winter, she rescued a three-legged dog named Max from an abandoned warehouse. Max had been severely injured, and Emma spent months rehabilitating him. Despite his disability, Max became a symbol of resilience in the community, helping other injured animals and inspiring local children about compassion and overcoming challenges.

    Emma's rescue center was funded entirely by local donations and her own savings. She worked tirelessly, often staying up all night to care for injured animals. Her most memorable rescue was Max, who not only recovered but became a therapy dog, visiting schools and hospitals to spread hope and awareness about animal welfare.
    """
    
    # Display the story
    st.subheader("Sample Story")
    st.write(sample_story)
    
    # Instantiate the RAG application
    rag_app = RAGComparison()
    
    # Chunk and embed text
    chunks = rag_app.chunk_text(sample_story)
    embeddings = rag_app.create_embeddings(chunks)
    
    # Predefined questions
    questions = [
        "Who is Emma Johnson?",
        "What happened to Max?",
        "How does Max help the community?"
    ]
    
    # Question Selection
    query = st.selectbox("Choose a question about the story", questions)
    
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
        "1. Read the sample story\n"
        "2. Select a question\n"
        "3. Compare Zero-Shot vs RAG responses"
    )

if __name__ == "__main__":
    main()