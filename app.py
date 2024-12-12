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
    Titanic Survival Prediction Model Report
        Data Preprocessing
            The dataset underwent careful preprocessing, which involved handling missing values, encoding categorical variables, and normalizing numerical features to prepare the data for machine learning analysis.
        Model: XGBoost Classifier
            Features and Importance
            The XGBoost model revealed a clear hierarchy of feature importance in predicting passenger survival. Fare emerged as the most critical feature, accounting for the highest importance score of 0.352. This was closely followed by Age, which contributed 0.276 to the model's predictive power. Gender (Sex) ranked third with an importance score of 0.215, demonstrating its significant impact on survival chances. Passenger class (Pclass) contributed 0.097 to the model, while the embarkation point (Embarked) played a minor role with 0.038. Familial connections through Sibling/Spouse (SibSp) and Parent/Child (Parch) relationships had the least influence, with scores of 0.022 and 0.010 respectively.
         Model Performance Metrics
            The XGBoost classifier demonstrated robust performance across multiple evaluation metrics. The model achieved an overall accuracy of 85%, with precision and recall both hovering around 0.83. The F1 score of 0.83 indicates a balanced performance between precision and recall, while the AUC-ROC score of 0.89 suggests strong predictive capabilities in distinguishing between survival and non-survival scenarios.
        Key Insights
            Fare and passenger class emerged as the most critical predictors of survival, indicating the significant role of socioeconomic factors during the Titanic disaster. Gender played a crucial role in survival probability, with clear disparities in rescue rates. Age was found to be the second most important feature, suggesting that a passenger's age significantly influenced their chances of survival.
        Recommendations
            To enhance the model's predictive power, researchers should consider incorporating more detailed passenger information. Exploring ensemble methods could potentially improve model performance. Additionally, validating the model against additional historical datasets would provide further robustness to the analysis.

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