# Financial Analyst Agent
A sophisticated AI-powered financial analysis system that combines Retrieval-Augmented Generation (RAG), real-time web search, and advanced financial calculations to provide comprehensive investment insights and company analysis.

## 🌟 Features
- **Multi-Tool Architecture**: Intelligent orchestration of RAG, web search, and calculation tools
- **Natural Language Processing**: Parse complex financial queries in plain English
- **Historical Data Analysis**: RAG-powered retrieval from 10-K filings and annual reports
- **Real-Time Market Data**: Current stock prices, market cap, and trading information
- **Financial Calculations**: Automated computation of P/E ratios, profit margins, growth rates, and more
- **Smart Query Routing**: Automatically determines which tools to use based on query requirements

## 🏗️ Architecture
- The system consists of four main components:

### 1. Query Parser
- Extracts company names, financial metrics, and periods from natural language
- Determines whether current data, historical data, or calculations are needed
- Converts unstructured queries into structured `FinancialQuery` objects

### 2. RAG Tool
- Processes and indexes financial documents using sentence transformers
- Uses FAISS for efficient similarity search
- Retrieves relevant historical financial information from 10-K filings and annual reports
- Supports both full-featured mode (with dependencies) and demo mode

### 3. Web Search Tool
- Fetches real-time stock prices and market data
- Integrates with Alpha Vantage API (with mock data fallback)
- Provides current trading information, volume, and price changes
- Includes stock symbol resolution for company names

### 4. Calculator Tool
- Performs complex financial calculations and ratio analysis
- Computes P/E ratios, profit margins, market capitalization, and growth rates
- Extracts metrics from multiple data sources intelligently
- Provides formatted financial insights

## 📊 Sample Data

The system includes comprehensive 2024-2025 financial data for demonstration:

- **Apple Inc.**: $412.8B revenue (2025), $6.45 EPS, 47.1% gross margin
- **Microsoft Corp.**: $278.4B revenue (2025), 20% cloud growth, $98.3B net income
- **Tesla Inc.**: $118.7B revenue (2025), 22.6% growth rate, 2.15M vehicle deliveries

## 🚀 Quick Start
### Installation
```bash
# Clone the repository
git clone <repository-url>
cd financial-analyst-agent

# Install required dependencies
pip install pandas sentence-transformers faiss-cpu beautifulsoup4 PyPDF2 numpy requests

# Optional: Set up Alpha Vantage API key for real-time data
export ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

### Basic Usage

```python
from main import FinancialAnalystAgent

# Initialize the agent
agent = FinancialAnalystAgent()

# Load your financial documents
sample_filings = [
    {
        "company": "Apple",
        "year": "2025",
        "content": "Apple Inc. reported total net sales of $412.8 billion..."
    }
]

agent.load_financial_documents(sample_filings)

# Ask questions in natural language
result = agent.answer_query("What was Apple's revenue growth from 2024 to 2025?")
print(result["analysis"])
```

### Demo Mode
- If you don't have all dependencies installed, the system runs in demo mode:

```bash
python main.py
```

Or view the HTML demo:
```bash
open index.html
```

## 💡 Example Queries
- The system can handle various types of financial queries:
- **Historical Analysis**: "What was Apple's revenue last year?"
- **Current Market Data**: "What is Microsoft's current stock price?"
- **Comparative Analysis**: "Compare Tesla's 2024 and 2025 revenue"
- **Financial Calculations**: "Calculate Apple's P/E ratio based on latest earnings"
- **Growth Analysis**: "What is Microsoft's cloud revenue growth rate?"
- **Multi-metric Queries**: "Show me Apple's revenue, profit margin, and current valuation"

## 🔧 Configuration
### Environment Variables
```bash
ALPHA_VANTAGE_API_KEY=your_api_key_here  # For real-time stock data
```

### Custom Document Loading
```python
# Load your own financial documents
custom_filings = [
    {
        "company": "YourCompany",
        "year": "2025",
        "content": "Financial statement content here..."
    }
]

agent.load_financial_documents(custom_filings)
```

## 📈 Supported Financial Metrics

- **Revenue Metrics**: Total revenue, net sales, segment revenue
- **Profitability**: Net income, profit margins, operating income
- **Market Valuation**: Stock price, market cap, P/E ratio, EPS
- **Growth Rates**: Year-over-year revenue growth, earnings growth
- **Operational**: Cash flow, shares outstanding, R&D expenses

## 🔍 Query Examples and Results

### Simple Revenue Query
```
Query: "What was Apple's revenue last year?"
Result: Apple reported $412.8 billion in revenue for 2025, up from $391.0 billion in 2024
```

### Complex Multi-Tool Query
```
Query: "Calculate Apple's P/E ratio and compare it to the industry average."
Tools Used: RAG + WebSearch + Calculator
Result: P/E ratio of 30.34 based on EPS of $6.45 and current stock price of $195.67
```

## 📁 Project Structure

```
financial-analyst-agent/
├── main.py                 # Main agent implementation
├── index.html             # Demo visualization
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🔧 Dependencies
### Required
- `requests` - API calls and web requests
- `python-dateutil` - Date parsing and manipulation

### Optional (for full functionality)
- `pandas` - Data manipulation and analysis
- `sentence-transformers` - Text embeddings for RAG
- `faiss-cpu` - Efficient similarity search
- `beautifulsoup4` - HTML parsing
- `PyPDF2` - PDF document processing
- `numpy` - Numerical computations

### Installation
```bash
pip install requests python-dateutil

# For full functionality
pip install pandas sentence-transformers faiss-cpu beautifulsoup4 PyPDF2 numpy
```

## 🌐 API Integration
### Alpha Vantage (Stock Data)
1. Sign up at [Alpha Vantage](https://www.alphavantage.co/)
2. Get your free API key
3. Set the environment variable: `ALPHA_VANTAGE_API_KEY=your_key`

### Alternative APIs
- Yahoo Finance API
- Financial Modeling Prep
- IEX Cloud
- Quandl

## 🧪 Testing

Run the built-in demo:
```bash
python main.py
```

View sample outputs in browser:
```bash
open index.html
```

## 🔮 Future Enhancements
- **Enhanced NLP**: More sophisticated query parsing with spaCy or transformers
- **Additional Data Sources**: Integration with SEC EDGAR, Bloomberg, Reuters
- **Advanced Calculations**: DCF modeling, technical analysis, risk metrics
- **Visualization**: Interactive charts and graphs for financial data
- **Export Capabilities**: PDF reports, Excel dashboards
- **Real-time Alerts**: Price movement notifications and earnings alerts

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer
- This tool is for educational and research purposes only. Always verify financial data with official sources before making investment decisions.

## 📄 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support
- For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation in the code comments
- Review the demo examples in `main.py` and `index.html`

## 📊 Performance Notes
- **RAG Performance**: Processes ~1000 document chunks efficiently
- **API Limits**: Alpha Vantage free tier: 5 requests/minute, 500/day
- **Memory Usage**: ~200MB for typical document sets
- **Response Time**: 1-3 seconds for most queries

---

**Built with ❤️ for financial analysis and AI-powered insights**
