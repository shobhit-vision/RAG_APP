import os
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
import json
from datetime import datetime
from typing_extensions import Concatenate
from pprint import pprint

# LangChain imports
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize AstraDB and embeddings with proper table setup
def init_astra():
    try:
        # Get AstraDB credentials from environment or user input
        astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        astra_db_id = os.getenv("ASTRA_DB_ID")
        
        if not astra_token or not astra_db_id:
            st.sidebar.warning("🔑 AstraDB credentials not found")
            with st.sidebar.expander("AstraDB Configuration"):
                astra_token = st.text_input("AstraDB Application Token", type="password")
                astra_db_id = st.text_input("AstraDB Database ID")
        
        if astra_token and astra_db_id:
            import cassio
            # Initialize cassio with AstraDB credentials
            cassio.init(token=astra_token, database_id=astra_db_id)
            
            # Initialize embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name, 
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize vector store with proper configuration
            astra_vector_store = Cassandra(
                embedding=embeddings,
                table_name="compliance_documents",
                keyspace=None,  # Let it use default keyspace
                session=None,
            )
            
            st.sidebar.success("✅ AstraDB connected successfully")
            return astra_vector_store, embeddings
        else:
            return None, None
    except Exception as e:
        st.error(f"Error initializing AstraDB: {str(e)}")
        st.info("""
        💡 **Troubleshooting Tips:**
        1. Make sure your AstraDB token and database ID are correct
        2. Ensure the database is active and accessible
        3. Try using a different table name
        4. Check if your AstraDB instance supports vector search
        """)
        return None, None

# Alternative initialization with manual table creation
def init_astra_with_table_creation():
    try:
        astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        astra_db_id = os.getenv("ASTRA_DB_ID")
        
        if not astra_token or not astra_db_id:
            return None, None
            
        import cassio
        from cassandra.cluster import Cluster
        from cassandra.auth import PlainTextAuthProvider
        import cassandra
        
        # Initialize cassio
        cassio.init(token=astra_token, database_id=astra_db_id)
        
        # Initialize embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, 
            model_kwargs={'device': 'cpu'}
        )
        
        # Create a custom table with proper schema
        table_name = "compliance_docs_v1"
        
        # Initialize vector store
        astra_vector_store = Cassandra(
            embedding=embeddings,
            table_name=table_name,
            keyspace=None,
            session=None,
        )
        
        return astra_vector_store, embeddings
        
    except Exception as e:
        st.error(f"Error in AstraDB setup: {str(e)}")
        return None, None

# Simple text storage without vector search (fallback)
class SimpleTextStore:
    def __init__(self):
        self.documents = []
        self.metadata = []
    
    def add_texts(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        for text, metadata in zip(texts, metadatas):
            self.documents.append(Document(page_content=text, metadata=metadata))
        
        return len(texts)
    
    def similarity_search(self, query, k=5):
        # Simple keyword-based search as fallback
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            score = sum(1 for word in query_words if word in doc.page_content.lower())
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]
    
    def get_document_count(self):
        return len(self.documents)

# Initialize Groq client
def get_groq_api():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.sidebar.warning("🔑 Groq API Key not found")
        api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
    
    return api_key

# Text extraction functions
def extract_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def extract_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        st.error(f"Error extracting from URL: {str(e)}")
        return None

def extract_from_txt(file):
    try:
        text = file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error extracting text file: {str(e)}")
        return None

# Process and chunk text
def process_text_for_astra(text, astra_vector_store, source_info):
    try:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=400,
            chunk_overlap=100,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(text)
        
        # Add metadata to each chunk
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks[:50]):
            metadata = {
                "source": source_info["type"],
                "source_name": source_info.get("name", "Unknown"),
                "chunk_id": i,
                "timestamp": datetime.now().isoformat()
            }
            texts.append(chunk)
            metadatas.append(metadata)
        
        # Add to vector store
        if texts:
            # Use add_texts method which is more reliable
            astra_vector_store.add_texts(texts=texts, metadatas=metadatas)
            return len(texts)
        return 0
    except Exception as e:
        st.error(f"Error processing text for AstraDB: {str(e)}")
        return 0

# Initialize LLM
def init_llm(api_key):
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Format documents for retrieval
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Clause detection function
def detect_clauses(query, astra_vector_store, llm):
    try:
        # Search for relevant clauses
        relevant_docs = astra_vector_store.similarity_search(query, k=5)
        
        if not relevant_docs:
            return "No relevant clauses found in the stored documents."
        
        # Format context from retrieved documents
        context = format_docs(relevant_docs)
        
        # Create prompt for clause detection
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a compliance expert. Analyze the following documents and identify clauses related to the user's query. 
             For each relevant clause found:
             1. Clearly state which document it comes from
             2. Quote the exact clause text
             3. Explain its relevance to the query
             4. Assess compliance level (Fully Compliant, Partially Compliant, Non-Compliant)
             5. Provide specific recommendations for improvement
             
             Documents context:
             {context}"""),
            ("human", "Query: {query}")
        ])
        
        # Create processing chain
        clause_chain = (
            {"context": lambda x: context, "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return clause_chain.invoke(query)
        
    except Exception as e:
        return f"Error detecting clauses: {str(e)}"

# Modern UI components
def styled_header(title, icon):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; display: flex; align-items: center;">
            <span style="font-size: 3rem; margin-right: 15px;">{icon}</span>
            {title}
        </h1>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            AI-Powered Contract Compliance Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Compliance Checker",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize storage with fallback
    astra_vector_store = None
    embeddings = None
    use_fallback = False
    
    # Try AstraDB first
    astra_vector_store, embeddings = init_astra()
    
    # If AstraDB fails, use fallback
    if not astra_vector_store:
        st.warning("⚠️ AstraDB connection failed. Using local text storage (limited functionality).")
        astra_vector_store = SimpleTextStore()
        use_fallback = True
    
    # Initialize Groq
    groq_api_key = get_groq_api()
    llm = init_llm(groq_api_key) if groq_api_key else None
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 AI Compliance Checker")
        st.markdown("---")
        
        # Status indicators
        st.subheader("🔧 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if not use_fallback:
                st.success("✅ AstraDB")
            else:
                st.warning("⚠️ Local Storage")
        
        with col2:
            if llm:
                st.success("✅ Groq API")
            else:
                st.error("❌ Groq API")
        
        if use_fallback:
            st.markdown("""
            <div class="warning-box">
            <strong>Local Storage Mode:</strong>
            <br>• Basic text search only
            <br>• No vector similarity
            <br>• Data persists only during session
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio("Navigate to", [
            "📊 Dashboard", 
            "📥 Upload Documents", 
            "🔍 Clause Detection", 
            "📋 Compliance Analysis"
        ])
        
        # Document count in sidebar
        st.markdown("---")
        if hasattr(astra_vector_store, 'get_document_count'):
            doc_count = astra_vector_store.get_document_count()
        else:
            # For AstraDB, we'll show a sample count
            try:
                sample_docs = astra_vector_store.similarity_search("", k=10)
                doc_count = len(sample_docs)
            except:
                doc_count = 0
                
        st.metric("Documents Stored", doc_count)

    # Main content based on selected page
    if page == "📊 Dashboard":
        render_dashboard(astra_vector_store, llm, use_fallback)
    elif page == "📥 Upload Documents":
        render_upload_documents(astra_vector_store, embeddings, use_fallback)
    elif page == "🔍 Clause Detection":
        render_clause_detection(astra_vector_store, llm, use_fallback)
    elif page == "📋 Compliance Analysis":
        render_compliance_analysis(astra_vector_store, llm, use_fallback)

def render_dashboard(astra_vector_store, llm, use_fallback=False):
    styled_header("AI Compliance Checker Dashboard", "🔍")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to AI-Powered Regulatory Compliance Checker
        
        This intelligent system helps you:
        - 📥 **Upload** contracts from multiple sources (PDF, URLs, Text)
        - 🔍 **Detect** specific clauses in your documents
        - 📋 **Analyze** compliance with regulations
        - ⚡ **Get instant insights** using AI
        
        **Supported Regulations:**
        - GDPR (Data Privacy)
        - HIPAA (Healthcare)
        - SOX (Financial)
        - CCPA (California Privacy)
        - PCI DSS (Payment Security)
        - And many more...
        """)
        
        if use_fallback:
            st.warning("""
            ⚠️ **Local Storage Mode Active**
            You're currently using local text storage. For full functionality:
            - Configure AstraDB credentials in the sidebar
            - Enable vector similarity search
            - Get persistent storage across sessions
            """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        
        1. **Upload Documents** - Add your contracts
        2. **Detect Clauses** - Find specific provisions
        3. **Analyze Compliance** - Get AI-powered insights
        4. **Download Reports** - Share findings
        """)
    
    st.markdown("---")
    
    # Feature highlights
    st.subheader("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Multi-Format Support
        - PDF Documents
        - Web URLs
        - Text Files
        - Direct Text Input
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Smart Analysis
        - Clause Detection
        - Compliance Scoring
        - Risk Identification
        - Recommendation Engine
        """)
    
    with col3:
        if not use_fallback:
            st.markdown("""
            ### 💾 Intelligent Storage
            - Vector Database
            - Semantic Search
            - Context Understanding
            - Fast Retrieval
            """)
        else:
            st.markdown("""
            ### 💾 Basic Storage
            - Local Text Storage
            - Keyword Search
            - Session Persistence
            - Fast Setup
            """)

def render_upload_documents(astra_vector_store, embeddings, use_fallback=False):
    styled_header("Document Upload Center", "📥")
    
    if use_fallback:
        st.info("📝 **Local Storage Mode**: Documents will be stored in memory during this session only.")
    
    st.markdown("""
    Upload your contracts and documents in multiple formats. The AI system will process and store them 
    for intelligent compliance analysis and clause detection.
    """)
    
    # Upload options
    upload_option = st.radio(
        "Select upload method:",
        ["PDF File", "URL", "Text File", "Direct Text Input"],
        horizontal=True
    )
    
    extracted_content = None
    source_info = {}
    
    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'])
        if uploaded_file:
            with st.spinner("📄 Extracting text from PDF..."):
                extracted_content = extract_from_pdf(uploaded_file)
            source_info = {"type": "PDF", "name": uploaded_file.name}
    
    elif upload_option == "URL":
        url = st.text_input("Enter document URL:", placeholder="https://example.com/contract.pdf")
        if url:
            if st.button("🌐 Extract from URL"):
                with st.spinner("🔄 Fetching and extracting content..."):
                    extracted_content = extract_from_url(url)
                source_info = {"type": "URL", "name": url}
    
    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose text file", type=['txt'])
        if text_file:
            with st.spinner("📝 Reading text file..."):
                extracted_content = extract_from_txt(text_file)
            source_info = {"type": "TXT", "name": text_file.name}
    
    else:  # Direct Text Input
        direct_text = st.text_area(
            "Paste your contract text:",
            height=200,
            placeholder="Paste your contract clauses, terms, or compliance text here..."
        )
        if direct_text:
            extracted_content = direct_text
            source_info = {"type": "Direct Input", "name": "User Input"}
    
    # Process extracted content
    if extracted_content:
        st.success("✅ Content extracted successfully!")
        
        # Display preview
        with st.expander("📋 Content Preview", expanded=True):
            preview_text = extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content
            st.text_area("Preview", preview_text, height=200, label_visibility="collapsed")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", len(extracted_content))
        with col2:
            st.metric("Words", len(extracted_content.split()))
        with col3:
            st.metric("Lines", len(extracted_content.splitlines()))
        with col4:
            st.metric("Source", source_info["type"])
        
        # Store in database
        if st.button("💾 Store Documents", use_container_width=True):
            with st.spinner("🔄 Processing and storing documents..."):
                chunk_count = process_text_for_astra(extracted_content, astra_vector_store, source_info)
            
            if chunk_count > 0:
                st.success(f"✅ Successfully stored {chunk_count} text chunks!")
                if use_fallback:
                    st.info("💡 Documents are stored in local memory for this session only.")
            else:
                st.error("❌ Failed to store content")
        
        # Download option
        st.download_button(
            label="📥 Download Extracted Text",
            data=extracted_content,
            file_name=f"extracted_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def render_clause_detection(astra_vector_store, llm, use_fallback=False):
    styled_header("Smart Clause Detection", "🔍")
    
    if use_fallback:
        st.info("🔍 **Local Search Mode**: Using keyword-based search (limited to exact matches).")
    
    if not llm:
        st.error("🚫 Groq API key required for clause detection. Please configure it in the sidebar.")
        return
    
    st.markdown("""
    Enter a clause or compliance requirement you want to detect in your stored documents. 
    The AI will search through all uploaded contracts and identify relevant clauses.
    """)
    
    # Clause query input
    query = st.text_area(
        "What clause are you looking for?",
        height=100,
        placeholder="e.g., 'data privacy clause', 'termination conditions', 'confidentiality agreement', 'GDPR compliance requirements'..."
    )
    
    if st.button("🔍 Detect Clauses", use_container_width=True) and query:
        with st.spinner("🔄 Analyzing documents for relevant clauses..."):
            results = detect_clauses(query, astra_vector_store, llm)
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Detected Clauses")
        
        if "No relevant clauses" in results:
            st.warning("❌ No relevant clauses found matching your query.")
            if use_fallback:
                st.info("💡 Try using more specific keywords in local storage mode.")
        else:
            st.success("✅ Found relevant clauses!")
            st.markdown(results)
            
            # Download option
            st.download_button(
                label="💾 Download Analysis",
                data=results,
                file_name=f"clause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

def render_compliance_analysis(astra_vector_store, llm, use_fallback=False):
    styled_header("Compliance Analysis", "📋")
    
    if use_fallback:
        st.info("📊 **Local Analysis Mode**: Using available documents for compliance checking.")
    
    if not llm:
        st.error("🚫 Groq API key required for compliance analysis. Please configure it in the sidebar.")
        return
    
    st.markdown("""
    Analyze your contracts for regulatory compliance. Select a regulation or enter a custom compliance requirement.
    """)
    
    # Compliance options
    analysis_type = st.selectbox(
        "Select analysis type:",
        [
            "GDPR Compliance",
            "HIPAA Compliance", 
            "SOX Compliance",
            "CCPA Compliance",
            "PCI DSS Compliance",
            "Custom Compliance Check"
        ]
    )
    
    if analysis_type == "Custom Compliance Check":
        custom_query = st.text_area(
            "Enter specific compliance requirements:",
            height=100,
            placeholder="e.g., 'Check for data retention policies', 'Analyze security breach notification clauses'..."
        )
        query = custom_query
    else:
        regulation_map = {
            "GDPR Compliance": "General Data Protection Regulation data privacy and protection requirements",
            "HIPAA Compliance": "Health Insurance Portability and Accountability Act healthcare data security",
            "SOX Compliance": "Sarbanes-Oxley Act financial reporting and internal controls", 
            "CCPA Compliance": "California Consumer Privacy Act consumer data rights",
            "PCI DSS Compliance": "Payment Card Industry Data Security Standard payment card security"
        }
        query = regulation_map[analysis_type]
        st.info(f"🔍 Analyzing for {analysis_type} requirements")
    
    if st.button("🚀 Run Compliance Analysis", use_container_width=True) and query:
        with st.spinner("🔍 Conducting comprehensive compliance analysis..."):
            compliance_results = detect_clauses(query, astra_vector_store, llm)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Compliance Analysis Results")
        
        if compliance_results:
            st.markdown(compliance_results)
            
            # Download option
            st.download_button(
                label="💾 Download Full Report",
                data=compliance_results,
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
