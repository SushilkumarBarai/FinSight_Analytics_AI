import streamlit as st
import numpy as np
import re
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import json

# Import LangChain components with error handling
try:
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.error(f"LangChain components not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Initialize session state for multi-user support
def initialize_session_state():
    """Initialize session state variables for each user session"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
    
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

def clean_temp_files(user_id):
    """Clean up temporary files for a specific user"""
    temp_dir = Path(tempfile.gettempdir())
    for temp_file in temp_dir.glob(f"financial_report_{user_id}*"):
        try:
            temp_file.unlink()
        except Exception:
            pass

def process_document(uploaded_file, model_choice, user_id):
    """Process the uploaded document and create QA system"""
    try:
        # Create a unique temporary file for this user session
        temp_file_path = os.path.join(tempfile.gettempdir(), f"financial_report_{user_id}.pdf")
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load and process document
        loader = PDFPlumberLoader(temp_file_path)
        docs = loader.load()
        
        if not docs:
            st.error("No content could be extracted from the PDF. Please check if the file is valid.")
            return None
        
        # Document splitting and embedding
        progress_bar = st.progress(0)
        st.text("Creating document chunks...")
        
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        documents = text_splitter.split_documents(docs)
        progress_bar.progress(30)
        
        st.text("Creating embeddings...")
        embedder = HuggingFaceEmbeddings()
        vector = FAISS.from_documents(documents, embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        progress_bar.progress(70)
        
        st.text("Setting up language model...")
        llm = Ollama(model=model_choice)
        
        # Enhanced financial analysis prompt
        financial_prompt = """
        You are an expert financial analyst. Analyze the provided financial document context and answer the question with a structured, professional response.

        ### **Analysis Guidelines:**
        - Provide specific, data-driven insights
        - Use clear section headings
        - Include relevant financial figures with proper formatting
        - Highlight key trends and patterns
        - Offer actionable recommendations when appropriate
        - Keep responses concise but comprehensive (300-500 words)

        ### **Context from Financial Document:**
        {context}

        ### **Question:**
        {question}

        ### **Structured Financial Analysis:**
        """
        
        QA_PROMPT = PromptTemplate.from_template(financial_prompt)
        llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain, 
            document_variable_name="context"
        )
        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain, 
            retriever=retriever
        )
        
        progress_bar.progress(100)
        st.success("Document processed successfully!")
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
        
        return qa
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        # Clean up on error
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
        return None

def format_financial_response(response):
    """Clean and format the financial analysis response"""
    # Remove thinking artifacts
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    clean_response = re.sub(r"<thinking>.*?</thinking>", "", clean_response, flags=re.DOTALL)
    
    # Clean up extra whitespace
    clean_response = re.sub(r'\n\s*\n', '\n\n', clean_response)
    clean_response = clean_response.strip()
    
    # Format currency and numbers (basic formatting)
    clean_response = re.sub(r'\$(\d+)([KMB])', r'$\1 \2', clean_response)
    clean_response = re.sub(r'(\d+)%', r'\1%', clean_response)
    
    return clean_response

def export_analysis_history(analysis_history, file_name):
    """Export analysis history as JSON"""
    export_data = {
        "file_name": file_name,
        "export_timestamp": datetime.now().isoformat(),
        "analysis_history": analysis_history
    }
    return json.dumps(export_data, indent=2)

# Streamlit UI setup
st.set_page_config(
    page_title="Financial Report Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Main title with user session info
st.title("📊 Financial Report Analysis Platform")
st.caption(f"Session ID: {st.session_state.user_id[:8]}...")

if not LANGCHAIN_AVAILABLE:
    st.error("Required dependencies are not installed. Please install langchain and related packages.")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection with validation
    available_models = ["deepseek-r1:1.5b", "llama3.2", "llama2", "mistral"]
    model_choice = st.selectbox(
        "Select LLM Model", 
        available_models, 
        help="Choose the language model for analysis"
    )
    
    # Analysis focus areas
    analysis_types = st.multiselect(
        "Analysis Focus Areas", 
        [
            "Financial Metrics", 
            "Risk Analysis", 
            "Market Analysis", 
            "Performance Indicators", 
            "Cash Flow Analysis", 
            "Balance Sheet Analysis", 
            "Ratio Analysis",
            "Strategic Analysis"
        ], 
        default=["Financial Metrics", "Risk Analysis"]
    )
    
    st.divider()
    
    # Session management
    st.header("📋 Session Management")
    if st.session_state.current_file_name:
        st.info(f"Current document: {st.session_state.current_file_name}")
        st.write(f"Analyses performed: {len(st.session_state.analysis_history)}")
    
    if st.button("🗑️ Clear Session", type="secondary"):
        # Clear session state
        for key in ['qa_system', 'document_processed', 'analysis_history', 'current_file_name']:
            if key in st.session_state:
                del st.session_state[key]
        clean_temp_files(st.session_state.user_id)
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload Financial Report (PDF)", 
        type="pdf",
        help="Upload a PDF financial report for analysis"
    )
    
    if uploaded_file and (not st.session_state.document_processed or 
                         st.session_state.current_file_name != uploaded_file.name):
        
        st.info(f"Processing: {uploaded_file.name}")
        
        with st.spinner("Processing document... This may take a few minutes."):
            qa_system = process_document(uploaded_file, model_choice, st.session_state.user_id)
            
            if qa_system:
                st.session_state.qa_system = qa_system
                st.session_state.document_processed = True
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.analysis_history = []  # Reset history for new document

with col2:
    st.header("💡 Interactive Analysis")
    
    if st.session_state.document_processed and st.session_state.qa_system:
        # Suggested questions based on analysis types
        suggested_questions = {
            "Financial Metrics": [
                "What are the key revenue trends over the reporting periods?",
                "How has profitability evolved and what are the main drivers?",
                "What are the primary cost components and their changes?"
            ],
            "Risk Analysis": [
                "What are the main risk factors identified in the report?",
                "How is the company managing financial and operational risks?",
                "What regulatory or market risks are highlighted?"
            ],
            "Market Analysis": [
                "What is the company's competitive position in the market?",
                "What market opportunities and threats are identified?",
                "How do market conditions affect the business?"
            ],
            "Strategic Analysis": [
                "What are the company's strategic priorities and initiatives?",
                "How is the company positioning for future growth?",
                "What are the key strategic challenges mentioned?"
            ]
        }
        
        # Compile questions based on selected analysis types
        relevant_questions = []
        for analysis_type in analysis_types:
            if analysis_type in suggested_questions:
                relevant_questions.extend(suggested_questions[analysis_type])
        
        # Question selection
        selected_question = st.selectbox(
            "💭 Suggested Questions:", 
            [""] + relevant_questions,
            help="Select from pre-defined questions or enter your own below"
        )
        
        user_input = st.text_area(
            "✏️ Your Custom Question:", 
            value=selected_question,
            height=100,
            help="Enter your specific question about the financial report"
        )
        
        col2_1, col2_2 = st.columns([1, 1])
        
        with col2_1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col2_2:
            if st.session_state.analysis_history:
                export_button = st.button("📥 Export History", use_container_width=True)
            else:
                st.button("📥 Export History", disabled=True, use_container_width=True)
        
        # Perform analysis
        if analyze_button and user_input.strip():
            with st.spinner("Analyzing your question..."):
                try:
                    response = st.session_state.qa_system(user_input)["result"]
                    clean_response = format_financial_response(response)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    st.markdown(clean_response)
                    
                    # Add to history
                    analysis_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "question": user_input,
                        "response": clean_response
                    }
                    st.session_state.analysis_history.append(analysis_entry)
                    
                    st.success("Analysis completed!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        # Export functionality
        if st.session_state.analysis_history and 'export_button' in locals() and export_button:
            export_data = export_analysis_history(
                st.session_state.analysis_history, 
                st.session_state.current_file_name
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Analysis History",
                data=export_data,
                file_name=f"financial_analysis_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Display analysis history
        if st.session_state.analysis_history:
            st.subheader("📝 Analysis History")
            with st.expander("View Previous Analyses", expanded=False):
                for i, entry in enumerate(reversed(st.session_state.analysis_history), 1):
                    st.markdown(f"**Question {i}:** {entry['question']}")
                    st.markdown(f"**Analysis:** {entry['response'][:200]}...")
                    st.caption(f"Analyzed at: {entry['timestamp']}")
                    st.divider()
    
    else:
        st.info("👆 Please upload a financial report PDF to begin analysis.")
        st.markdown("""
        ### How to use this tool:
        1. **Upload** a PDF financial report
        2. **Select** analysis focus areas in the sidebar
        3. **Choose** a suggested question or write your own
        4. **Click** Analyze to get insights
        5. **Export** your analysis history when done
        """)

# Footer
st.markdown("---")
st.caption("🔒 Each user session is isolated and temporary files are automatically cleaned up.")