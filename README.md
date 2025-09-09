# 📊 Financial Report Analyzer

A powerful, multi-user Streamlit application that uses AI to analyze financial reports and provide intelligent insights. Built with LangChain and Ollama for advanced document processing and natural language analysis.

<p align="center">
  <img src="https://raw.githubusercontent.com/SushilkumarBarai/FinSight_Analytics_AI/main/Screenshot_1.png" alt="FinSight Analytics AI Screenshot" width="600"/>
</p>



## ✨ Features

- 🤖 **AI-Powered Analysis**: Leverages advanced language models for comprehensive financial analysis
- 📄 **PDF Processing**: Extract and analyze content from financial report PDFs
- 👥 **Multi-User Support**: Isolated sessions for concurrent users
- 🎯 **Focused Analysis**: Multiple analysis types including financial metrics, risk analysis, and market insights
- 💬 **Interactive Q&A**: Ask custom questions or use suggested prompts
- 📊 **Analysis History**: Track all your questions and responses
- 📥 **Export Functionality**: Download analysis results in JSON format
- 🔒 **Secure Sessions**: Each user gets isolated, temporary workspace

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- At least one language model downloaded in Ollama

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SushilkumarBarai/FinSight_Analytics_AI.git
   cd FinSight_Analytics_AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and setup Ollama**
   - Download Ollama from [https://ollama.ai](https://ollama.ai)
   - Install at least one model:
   ```bash
   ollama pull llama3.2
   ollama pull deepseek-r1:1.5b
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

Create a `requirements.txt` file with these dependencies:

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-experimental>=0.0.50
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
transformers>=4.35.0
torch>=2.1.0
pdfplumber>=0.9.0
numpy>=1.24.0
```

## 🎯 Usage Guide

### 1. **Upload Financial Report**
   - Click "Upload Financial Report (PDF)" 
   - Select your PDF financial document
   - Wait for processing (may take 2-3 minutes for large documents)

### 2. **Configure Analysis**
   - **Select LLM Model**: Choose from available Ollama models
   - **Analysis Focus Areas**: Select one or more analysis types:
     - Financial Metrics
     - Risk Analysis  
     - Market Analysis
     - Performance Indicators
     - Cash Flow Analysis
     - Balance Sheet Analysis
     - Ratio Analysis
     - Strategic Analysis

### 3. **Ask Questions**
   - **Suggested Questions**: Select from dynamically generated questions based on your analysis focus
   - **Custom Questions**: Write your own specific questions about the financial report
   - **Examples**:
     - "What are the key revenue trends over the past three years?"
     - "What are the main risk factors mentioned in the report?"
     - "How has the company's profitability changed?"

### 4. **Review Analysis**
   - Get structured, professional analysis responses
   - View analysis history for the current session
   - Export results for future reference

### 5. **Session Management**
   - Each user gets an isolated session
   - Clear session to start fresh
   - Automatic cleanup of temporary files

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Document        │───▶│   LangChain     │
│                 │    │  Processing      │    │   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         │              │   PDF Extraction │    │   Vector Store  │
         │              │   (PDFPlumber)   │    │    (FAISS)      │
         │              └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                            ┌─────────────────┐
│ Session State   │                            │     Ollama      │
│ Management      │                            │   LLM Models    │
└─────────────────┘                            └─────────────────┘
```

## 🔧 Configuration

### Ollama Models

Supported models (install via Ollama):
- `deepseek-r1:1.5b` - Fast, efficient model
- `llama3.2` - Balanced performance
- `llama2` - Reliable option
- `mistral` - Good for analysis

### Analysis Types

| Analysis Type | Description |
|---------------|-------------|
| **Financial Metrics** | Revenue, profit, expenses, key financial indicators |
| **Risk Analysis** | Market risks, operational risks, regulatory concerns |
| **Market Analysis** | Competitive position, market opportunities |
| **Performance Indicators** | KPIs, performance trends, benchmarks |
| **Cash Flow Analysis** | Cash flow patterns, liquidity analysis |
| **Balance Sheet Analysis** | Assets, liabilities, equity analysis |
| **Ratio Analysis** | Financial ratios and their interpretation |
| **Strategic Analysis** | Strategic initiatives, future outlook |

## 🛠️ Advanced Features

### Multi-User Support
- **Session Isolation**: Each user gets a unique session ID
- **Concurrent Processing**: Multiple users can analyze documents simultaneously
- **Resource Management**: Automatic cleanup prevents server overload

### Error Handling
- **Graceful Degradation**: Continues working even if some features fail
- **User-Friendly Messages**: Clear error messages and recovery suggestions
- **Validation**: Checks for required dependencies and valid files

### Export Options
- **JSON Format**: Structured export of analysis history
- **Timestamped**: Each analysis includes timestamp
- **Complete History**: All questions and responses in session

## 🚨 Troubleshooting

### Common Issues

**1. "LangChain components not available"**
```bash
pip install langchain langchain-community langchain-experimental
```

**2. "Model not found" error**
```bash
ollama pull llama3.2
ollama list  # Verify installed models
```

**3. "PDF processing failed"**
- Ensure PDF is not password-protected
- Check if PDF contains extractable text (not just images)
- Try with a different PDF file

**4. "Memory issues with large PDFs"**
- Use smaller PDF files (< 50MB)
- Restart the application
- Check available system memory

### Performance Tips

- **Use smaller models** (`deepseek-r1:1.5b`) for faster processing
- **Limit PDF size** to under 50MB for optimal performance
- **Close unused browser tabs** to free memory
- **Process one document at a time** per user session


## 📊 Example Use Cases

- **Investment Analysis**: Analyze annual reports for investment decisions
- **Due Diligence**: Review financial documents for M&A activities
- **Risk Assessment**: Identify potential financial and operational risks
- **Competitive Analysis**: Compare financial performance across companies
- **Regulatory Compliance**: Ensure compliance with financial reporting standards


**Made with ❤️ using Streamlit, LangChain, and Ollama**
