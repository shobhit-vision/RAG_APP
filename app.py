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
        astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        astra_db_id = os.getenv("ASTRA_DB_ID")

        if not astra_token or not astra_db_id:
            st.sidebar.warning("🔑 AstraDB credentials not found")
            with st.sidebar.expander("AstraDB Configuration"):
                astra_token = st.text_input("AstraDB Application Token", type="password")
                astra_db_id = st.text_input("AstraDB Database ID")

        if astra_token and astra_db_id:
            import cassio
            cassio.init(token=astra_token, database_id=astra_db_id)

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )

            astra_vector_store = Cassandra(
                embedding=embeddings,
                table_name="compliance_documents",
                keyspace=None,
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
def process_text_for_astra(text, astra_vector_store, source_info, doc_type="compliance"):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)

        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks[:50]):
            metadata = {
                "source": source_info["type"],
                "source_name": source_info.get("name", "Unknown"),
                "doc_type": doc_type,
                "chunk_id": i,
                "timestamp": datetime.now().isoformat()
            }
            texts.append(chunk)
            metadatas.append(metadata)

        if texts:
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
            temperature=0.1,
            max_tokens=8000
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Format documents for retrieval
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Retrieve relevant compliance documents using smart search
def retrieve_compliance_docs(contract_text, astra_vector_store, regulatory_focus):
    try:
        search_queries = []
        
        base_keywords = [
            "data protection", "privacy", "security", "confidentiality", 
            "liability", "indemnification", "warranty", "termination",
            "governing law", "intellectual property", "breach", "compliance"
        ]
        
        regulatory_queries = []
        for framework in regulatory_focus:
            if "GDPR" in framework:
                regulatory_queries.extend(["GDPR data protection", "personal data processing", "data subject rights"])
            if "HIPAA" in framework:
                regulatory_queries.extend(["HIPAA healthcare privacy", "protected health information", "PHI security"])
            if "SOX" in framework:
                regulatory_queries.extend(["SOX financial controls", "internal controls", "financial reporting"])
            if "CCPA" in framework:
                regulatory_queries.extend(["CCPA consumer privacy", "California privacy", "consumer rights"])
            if "PCI" in framework:
                regulatory_queries.extend(["PCI DSS security", "payment card security", "cardholder data"])
        
        all_queries = base_keywords + regulatory_queries
        
        all_docs = []
        for query in all_queries[:15]:
            try:
                docs = astra_vector_store.similarity_search(query, k=3)
                all_docs.extend(docs)
            except Exception as e:
                continue

        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)

        return unique_docs[:15]

    except Exception as e:
        st.error(f"Error retrieving compliance documents: {str(e)}")
        return []

# Single comprehensive compliance analysis function
def comprehensive_compliance_analysis(contract_text, compliance_docs, llm, regulatory_focus):
    try:
        compliance_context = format_docs(compliance_docs) if compliance_docs else "No specific compliance documents retrieved from database."

        regulatory_context = ", ".join(regulatory_focus)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a senior compliance analyst and legal expert. Perform a COMPREHENSIVE analysis of the contract against compliance requirements.

            COMPLIANCE REQUIREMENTS FROM DATABASE:
            {compliance_context}

            REGULATORY FOCUS:
            {regulatory_context}

            ANALYSIS INSTRUCTIONS:
            Perform a complete analysis in the following structured format:

            1. KEY CLAUSES IDENTIFIED: list all important clauses from the contract
            2. SPECIFIC ISSUES & VIOLATIONS: List exact compliance violations with references
            3. RECTIFIED CLAUSE SUGGESTIONS: Provide corrected versions of problematic clauses
            4. Parties detail: Like full legal names,addresses, contact information etc

            Be thorough, precise, and reference exact compliance requirements from the database.
            """),
            ("human", """Contract:\n{contract_text}""")
        ])

        analysis_chain = prompt | llm | StrOutputParser()
        return analysis_chain.invoke({
            "contract_text": contract_text
        })

    except Exception as e:
        return f"Error in comprehensive compliance analysis: {str(e)}"

# Local storage management for contracts
class ContractStorage:
    def __init__(self):
        self.contracts = {}
        self.next_id = 1

    def save_contract(self, contract_text, source_info):
        contract_id = f"contract_{self.next_id}"
        self.contracts[contract_id] = {
            "id": contract_id,
            "text": contract_text,
            "source_info": source_info,
            "upload_time": datetime.now().isoformat(),
            "name": source_info.get("name", f"Contract_{self.next_id}")
        }
        self.next_id += 1
        return contract_id

    def get_contract(self, contract_id):
        return self.contracts.get(contract_id)

    def get_all_contracts(self):
        return self.contracts

    def delete_contract(self, contract_id):
        if contract_id in self.contracts:
            del self.contracts[contract_id]
            return True
        return False

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
            AI-Powered Contract Compliance Analysis System
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Compliance Analyzer Pro",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    .contract-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 6px solid #667eea;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        color: #212529 !important;
    }
    .contract-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    .contract-card strong {
        color: #212529 !important;
        font-size: 1rem;
    }
    .contract-card small {
        color: #6c757d !important;
    }
    .compliance-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #38b2ac;
        color: #212529 !important;
    }
    .analysis-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
    }
    .contract-card.selected {
        border-left: 6px solid #28a745;
        background: linear-gradient(135deg, #e8f5e8 0%, #d1e7dd 100%);
    }
    .contract-card.selected strong {
        color: #155724 !important;
    }
    .contract-card.selected small {
        color: #0f5132 !important;
    }
    .analysis-result {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #212529 !important;
    }
    .analysis-result h1, .analysis-result h2, .analysis-result h3, .analysis-result h4, .analysis-result h5, .analysis-result h6 {
        color: #212529 !important;
    }
    .analysis-result p, .analysis-result li, .analysis-result div {
        color: #212529 !important;
    }
    .analysis-result strong {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    astra_vector_store, embeddings = init_astra()

    groq_api_key = get_groq_api()
    llm = init_llm(groq_api_key) if groq_api_key else None

    if 'contract_storage' not in st.session_state:
        st.session_state.contract_storage = ContractStorage()

    with st.sidebar:
        st.title("🔍 AI Compliance Analyzer Pro")
        st.markdown("---")

        st.subheader("🔧 System Status")

        col1, col2 = st.columns(2)
        with col1:
            if astra_vector_store:
                st.success("✅ AstraDB")
            else:
                st.error("❌ AstraDB")

        with col2:
            if llm:
                st.success("✅ Groq API")
            else:
                st.error("❌ Groq API")

        if not astra_vector_store:
            st.markdown("""
            <div class="error-box">
            <strong>AstraDB Required:</strong>
            <br>• Please configure AstraDB credentials
            <br>• Compliance upload requires vector database
            <br>• Contact admin for setup assistance
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio("Navigate to", [
            "🏠 Dashboard",
            "📥 Upload Compliance Docs",
            "📄 Upload & Analyze Contracts"
        ])

        st.markdown("---")
        contract_count = len(st.session_state.contract_storage.get_all_contracts())
        
        feedback = st.text_area("💬 Feedback", placeholder="Share your feedback...", height=80)
        if st.button("Submit Feedback", use_container_width=True):
            if feedback.strip():
                st.success("✅ Thank you for your feedback!")
        else:
            st.warning("⚠️ Please enter your feedback")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Contracts", contract_count)

        with col2:
          if contract_count > 0:
              st.subheader("📋 Stored Contracts")
              contracts = st.session_state.contract_storage.get_all_contracts()
              for contract_id, contract_data in contracts.items():
                  st.markdown(f"""
                  <div class="contract-card">
                  <strong>{contract_data['name']}</strong><br>
                  <small>Uploaded: {contract_data['upload_time'][:16]}</small>
                  </div>
                  """, unsafe_allow_html=True)

    if page == "🏠 Dashboard":
        render_dashboard(astra_vector_store, llm)
    elif page == "📥 Upload Compliance Docs":
        render_upload_compliance(astra_vector_store, embeddings)
    elif page == "📄 Upload & Analyze Contracts":
        render_upload_analyze_contracts(astra_vector_store, llm)

def render_dashboard(astra_vector_store, llm):
    styled_header("AI Powered Compliance Analyzer", "🏠")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""

        **🎯 Advanced Workflow:**
        1. **Upload Compliance Docs** - Store regulatory frameworks in AstraDB
        2. **Upload Contracts** - Add your contract documents
        3. **Smart Analysis** - Automated clause extraction + compliance checking
        4. **Precise Results** - Specific issues, risks, and recommendations

        **🛡️ Supported Regulatory Frameworks:**
        - **GDPR** - Data Privacy & Protection
        - **HIPAA** - Healthcare Information Security
        - **SOX** - Financial Reporting & Controls
        - **CCPA** - California Consumer Privacy
        - **PCI DSS** - Payment Card Security
        """)

        if not astra_vector_store:
            st.markdown("""
            <div class="error-box">
            ⚠️ **AstraDB Required**
            Compliance functionality requires AstraDB configuration. Please set up your AstraDB credentials in the sidebar.
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""

        🔍 **Comprehensive Analysis**
        - Single-step complete analysis
        - Full contract text processing
        - Smart compliance document retrieval
        - Complete risk assessment
        - Rectified clause suggestions

        ⚖️ **Detailed Reporting**
        - Executive summary
        - Key clauses identified
        - Compliance violations
        - Risk categorization
        - Actionable recommendations
        """)

    st.markdown("---")
    st.subheader("🚀 Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Upload Compliance Docs", use_container_width=True):
            st.session_state.current_page = "📥 Upload Compliance Docs"
            st.rerun()

    with col2:
        if st.button("📄 Upload & Analyze Contracts", use_container_width=True):
            st.session_state.current_page = "📄 Upload & Analyze Contracts"
            st.rerun()

def render_upload_compliance(astra_vector_store, embeddings):
    styled_header("Upload Compliance Documents", "📥")

    st.markdown("""
    ### 📚 Upload Regulatory Compliance Documents
    Upload your compliance frameworks, regulatory requirements, and standards documentation.
    These will be stored in AstraDB and used for smart contract analysis.
    """)

    if not astra_vector_store:
        st.markdown("""
        <div class="error-box">
        ❌ **AstraDB Not Configured**
        Compliance document upload requires AstraDB vector database. Please configure your AstraDB credentials in the sidebar.
        </div>
        """, unsafe_allow_html=True)
        return

    upload_option = st.radio(
        "Select upload method for compliance documents:",
        ["PDF File", "URL", "Text File", "Direct Text Input"],
        horizontal=True
    )

    extracted_content = None
    source_info = {}

    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose compliance PDF file", type=['pdf'], key="compliance_pdf")
        if uploaded_file:
            with st.spinner("📄 Extracting compliance text from PDF..."):
                extracted_content = extract_from_pdf(uploaded_file)
            source_info = {"type": "PDF", "name": uploaded_file.name}

    elif upload_option == "URL":
        url = st.text_input("Enter compliance document URL:", placeholder="https://example.com/gdpr-regulation.pdf", key="compliance_url")
        if url:
            if st.button("🌐 Extract Compliance from URL"):
                with st.spinner("🔄 Fetching and extracting compliance content..."):
                    extracted_content = extract_from_url(url)
                source_info = {"type": "URL", "name": url}

    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose compliance text file", type=['txt'], key="compliance_txt")
        if text_file:
            with st.spinner("📝 Reading compliance text file..."):
                extracted_content = extract_from_txt(text_file)
            source_info = {"type": "TXT", "name": text_file.name}

    else:
        direct_text = st.text_area(
            "Paste your compliance framework text:",
            height=200,
            placeholder="Paste GDPR, HIPAA, SOX, or other regulatory requirements here...",
            key="compliance_direct"
        )
        if direct_text:
            extracted_content = direct_text
            source_info = {"type": "Direct Input", "name": "Compliance Framework"}

    if extracted_content:
        st.markdown("""
        <div class="success-box">
        ✅ Compliance content extracted successfully!
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Compliance Content Preview", expanded=True):
            preview_text = extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content
            st.text_area("Preview", preview_text, height=200, label_visibility="collapsed", key="compliance_preview")

        if st.button("💾 Store Compliance Documents", use_container_width=True, key="store_compliance"):
            with st.spinner("🔄 Processing and storing compliance framework..."):
                chunk_count = process_text_for_astra(extracted_content, astra_vector_store, source_info, doc_type="compliance")

            if chunk_count > 0:
                st.markdown(f"""
                <div class="success-box">
                ✅ Successfully stored {chunk_count} compliance chunks in AstraDB!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                ❌ Failed to store compliance content
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    
def render_upload_analyze_contracts(astra_vector_store, llm):
    styled_header("Upload & Analyze Contracts", "📄")

    st.markdown("""
    ### 📄 Complete Contract Analysis Pipeline
    Upload contracts and run automated comprehensive compliance analysis in one seamless workflow.
    """)

    if 'current_contract_id' not in st.session_state:
        st.session_state.current_contract_id = None

    tab1, tab2 = st.tabs(["📥 Upload Contracts", "🔍 Smart Analysis"])

    with tab1:
        render_contract_upload_tab()

    with tab2:
        render_smart_analysis_tab(astra_vector_store, llm)

def render_contract_upload_tab():
    st.subheader("📥 Upload Contract Documents")

    upload_option = st.radio(
        "Select upload method:",
        ["PDF File", "URL", "Text File", "Direct Text Input"],
        horizontal=True,
        key="contract_upload"
    )

    extracted_content = None
    source_info = {}

    if upload_option == "PDF File":
        uploaded_file = st.file_uploader("Choose contract PDF file", type=['pdf'], key="contract_pdf")
        if uploaded_file:
            with st.spinner("📄 Extracting contract text from PDF..."):
                extracted_content = extract_from_pdf(uploaded_file)
            source_info = {"type": "PDF", "name": uploaded_file.name}

    elif upload_option == "URL":
        url = st.text_input("Enter contract document URL:", placeholder="https://example.com/service-agreement.pdf", key="contract_url")
        if url:
            if st.button("🌐 Extract Contract from URL", key="url_extract"):
                with st.spinner("🔄 Fetching and extracting contract content..."):
                    extracted_content = extract_from_url(url)
                source_info = {"type": "URL", "name": url}

    elif upload_option == "Text File":
        text_file = st.file_uploader("Choose contract text file", type=['txt'], key="contract_txt")
        if text_file:
            with st.spinner("📝 Reading contract text file..."):
                extracted_content = extract_from_txt(text_file)
            source_info = {"type": "TXT", "name": text_file.name}

    else:
        direct_text = st.text_area(
            "Paste your contract text:",
            height=200,
            placeholder="Paste your contract text, clauses, and agreements here...",
            key="contract_direct"
        )
        if direct_text:
            extracted_content = direct_text
            source_info = {"type": "Direct Input", "name": "Contract Document"}

    if extracted_content:
        st.markdown("""
        <div class="success-box">
        ✅ Contract content extracted successfully!
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Contract Content Preview", expanded=True):
            preview_text = extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content
            st.text_area("Preview", preview_text, height=200, label_visibility="collapsed", key="contract_preview")

        if st.button("💾 Save Contract to Storage", use_container_width=True, key="store_contract"):
            contract_id = st.session_state.contract_storage.save_contract(extracted_content, source_info)
            st.session_state.current_contract_id = contract_id
            st.markdown(f"""
            <div class="success-box">
            ✅ Contract saved successfully! (ID: {contract_id})
            </div>
            """, unsafe_allow_html=True)
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Stored Contracts")

    contracts = st.session_state.contract_storage.get_all_contracts()
    if not contracts:
        st.markdown("""
        <div class="info-box">
        ℹ️ No contracts stored yet. Upload a contract to get started.
        </div>
        """, unsafe_allow_html=True)
    else:
        for contract_id, contract_data in contracts.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{contract_data['name']}**")
                st.caption(f"Uploaded: {contract_data['upload_time'][:16]} | Characters: {len(contract_data['text'])}")
            with col2:
                if st.button("📝 Select", key=f"select_{contract_id}"):
                    st.session_state.current_contract_id = contract_id
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ Selected contract: {contract_data['name']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{contract_id}"):
                    if st.session_state.contract_storage.delete_contract(contract_id):
                        if st.session_state.current_contract_id == contract_id:
                            st.session_state.current_contract_id = None
                        st.markdown("""
                        <div class="success-box">
                        ✅ Contract deleted successfully!
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()

def render_smart_analysis_tab(astra_vector_store, llm):
    st.subheader("🔍 Smart Contract Analysis")

    if not llm:
        st.markdown("""
        <div class="error-box">
        ❌ Groq API key required for analysis. Please configure it in the sidebar.
        </div>
        """, unsafe_allow_html=True)
        return

    contracts = st.session_state.contract_storage.get_all_contracts()
    if not contracts:
        st.markdown("""
        <div class="warning-box">
        ⚠️ No contracts available. Please upload a contract first.
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        sample_compliance = astra_vector_store.similarity_search("compliance", k=1)
        has_compliance_docs = len(sample_compliance) > 0
    except:
        has_compliance_docs = False

    if not has_compliance_docs:
        st.markdown("""
        <div class="warning-box">
        ⚠️ No compliance documents found. Please upload compliance frameworks in the 'Upload Compliance Docs' section first.
        </div>
        """, unsafe_allow_html=True)
        if st.button("📥 Go to Upload Compliance Docs"):
            st.session_state.current_page = "📥 Upload Compliance Docs"
            st.rerun()
        return

    contract_names = {cid: data['name'] for cid, data in contracts.items()}
    selected_contract_id = st.selectbox(
        "Select contract for analysis:",
        options=list(contract_names.keys()),
        format_func=lambda x: contract_names[x],
        key="analysis_select"
    )

    if selected_contract_id:
        contract_data = st.session_state.contract_storage.get_contract(selected_contract_id)
        contract_text = contract_data['text']

        st.markdown(f"""
        <div class="success-box">
        ✅ Selected contract: **{contract_data['name']}** | Characters: {len(contract_text)}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🎯 Select Regulatory Frameworks")
        regulatory_focus = st.multiselect(
            "Choose frameworks to analyze against:",
            [
                "GDPR - Data Protection",
                "HIPAA - Healthcare Privacy",
                "SOX - Financial Controls",
                "CCPA - Consumer Privacy",
                "PCI DSS - Payment Security",
                "General Compliance"
            ],
            default=["GDPR - Data Protection", "General Compliance"]
        )

        if st.button("🚀 Run Comprehensive Analysis", use_container_width=True, type="primary"):
            if not regulatory_focus:
                st.markdown("""
                <div class="error-box">
                ❌ Please select at least one regulatory framework to analyze against.
                </div>
                """, unsafe_allow_html=True)
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📚 Step 1/2: Retrieving relevant compliance documents...")
                compliance_docs = retrieve_compliance_docs(contract_text, astra_vector_store, regulatory_focus)
                progress_bar.progress(50)

                with st.expander("📚 Retrieved Compliance Documents", expanded=False):
                    if compliance_docs:
                        st.success(f"✅ Retrieved {len(compliance_docs)} relevant compliance documents")
                        for i, doc in enumerate(compliance_docs[:5]):
                            st.markdown(f"**Document {i+1}** (Source: {doc.metadata.get('source_name', 'Unknown')})")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    else:
                        st.warning("⚠️ No specific compliance documents retrieved")

                status_text.text("⚖️ Step 2/2: Performing comprehensive compliance analysis...")
                analysis_result = comprehensive_compliance_analysis(
                    contract_text,
                    compliance_docs,
                    llm,
                    regulatory_focus
                )
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")

                st.markdown("---")
                st.markdown("""
                <div class="analysis-section">
                    <h2 style="color: white; margin: 0;">Comprehensive Compliance Analysis Results</h2>
                    <p style="color: white; margin: 0.5rem 0 0 0;">Complete assessment generated through smart pipeline</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="analysis-result">
                """ + analysis_result + """
                </div>
                """, unsafe_allow_html=True)

                full_report = f"""
COMPREHENSIVE COMPLIANCE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contract: {contract_data['name']}
Frameworks: {', '.join(regulatory_focus)}
Contract Length: {len(contract_text)} characters

ANALYSIS RESULTS:

{analysis_result}

                """

                st.download_button(
                    label="💾 Download Complete Analysis Report",
                    data=full_report,
                    file_name=f"comprehensive_compliance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error during comprehensive analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    main()
