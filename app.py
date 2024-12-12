import streamlit as st
import PyPDF2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGComparison:
    def __init__(self):
        # Initialize models with explicit configuration
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load tokenizer and model explicitly
        self.zero_shot_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.zero_shot_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Configure tokenizer
        self.zero_shot_tokenizer.pad_token = self.zero_shot_tokenizer.eos_token
        
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
        # Explicitly handle tokenization and generation
        inputs = self.zero_shot_tokenizer(
            query, 
            return_tensors="pt", 
            truncation=True,
            max_length=50
        )
        
        # Generate text
        outputs = self.zero_shot_model.generate(
            inputs.input_ids, 
            max_length=150, 
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=self.zero_shot_tokenizer.eos_token_id
        )
        
        # Decode the generated text
        response = self.zero_shot_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        return response
    
    def generate_rag_response(self, context, query):
        """Generate response with context using RAG"""
        result = self.rag_model({
            'context': context,
            'question': query
        })
        return result['answer']

def main():
    st.title("Multi-Mode RAG Comparison")
    
    # Sample Story
    sample_story = """
    In the small town of Riverdale, Emma Johnson was a dedicated local veterinarian who ran a unique animal rescue center. One winter, she rescued a three-legged dog named Max from an abandoned warehouse. Max had been severely injured, and Emma spent months rehabilitating him. Despite his disability, Max became a symbol of resilience in the community, helping other injured animals and inspiring local children about compassion and overcoming challenges.

    Emma's rescue center was funded entirely by local donations and her own savings. She worked tirelessly, often staying up all night to care for injured animals. Her most memorable rescue was Max, who not only recovered but became a therapy dog, visiting schools and hospitals to spread hope and awareness about animal welfare.
    """
    
    # Instantiate the RAG application
    rag_app = RAGComparison()
    
    # Input Method Selection
    input_mode = st.sidebar.radio(
        "Choose Input Mode", 
        ["Default Story", "Direct Text Input", "PDF Upload"]
    )
    
    # Text input based on mode
    if input_mode == "Default Story":
        st.subheader("Sample Story")
        st.write(sample_story)
        text = sample_story
    elif input_mode == "Direct Text Input":
        text = st.text_area(
            "Enter your text", 
            height=250, 
            placeholder="Paste the text you want to query..."
        )
    else:  # PDF Upload
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file is not None:
            text = rag_app.extract_text_from_pdf(uploaded_file)
            st.success("PDF Uploaded and Processed!")
        else:
            text = ""
    
    # Proceed if text is available
    if text:
        # Chunk and embed text
        chunks = rag_app.chunk_text(text)
        embeddings = rag_app.create_embeddings(chunks)
        
        # Predefined questions
        questions = st.text_input("Enter your question about the text")
        
        if questions:
            # Comparison Container
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Zero-Shot (Without RAG)")
                st.warning("Response without context")
                
                # Generate Zero-Shot Response
                zero_shot_response = rag_app.generate_zero_shot_response(questions)
                st.write(zero_shot_response)
            
            with col2:
                st.subheader("RAG Response")
                st.success("Response with context")
                
                # Find most relevant chunk
                relevant_chunk = rag_app.find_most_relevant_chunk(questions, chunks, embeddings)
                
                # Generate RAG Response
                rag_response = rag_app.generate_rag_response(relevant_chunk, questions)
                st.write(rag_response)
            
            # Show Relevant Context
            with st.expander("See Relevant Context"):
                st.write(relevant_chunk)

    # Additional information
    st.sidebar.markdown("### How to Use")
    st.sidebar.info(
        "1. Choose input mode\n"
        "2. Enter text or upload PDF\n"
        "3. Ask a question\n"
        "4. Compare Zero-Shot vs RAG responses"
    )

if __name__ == "__main__":
    main()