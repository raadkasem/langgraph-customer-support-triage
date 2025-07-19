# Intelligent Customer Support Triage System

A multi-agent system built with LangGraph that automates customer support query processing through intelligent classification, knowledge base retrieval, response generation, and escalation decision-making.

## 🎯 Project Overview

This system demonstrates advanced AI capabilities including:
- **Multi-Agent Orchestration** using LangGraph
- **RAG (Retrieval-Augmented Generation)** for knowledge base queries
- **Intelligent Query Classification** with LLM-powered categorization
- **Automated Escalation Logic** for complex queries
- **Customer Data Integration** with simulated CRM system

## 🏗️ System Architecture

The system consists of four main agents orchestrated by LangGraph:

1. **Query Classifier Agent** - Categorizes incoming queries
2. **Knowledge Base Agent (RAG)** - Retrieves relevant documentation
3. **Response Generation Agent** - Creates helpful responses
4. **Escalation Agent** - Determines if human intervention is needed

## 📁 Project Structure

```
langgraph-customer-support-triage/
├── agents/
│   ├── classifier_agent.py     # Query classification logic
│   ├── response_agent.py       # Response generation with tools
│   └── escalation_agent.py     # Escalation decision logic
├── tools/
│   ├── rag_tool.py            # Knowledge base search tool
│   └── crm_tool.py            # Customer data lookup tool
├── knowledge_base/
│   ├── password_reset.md      # Password reset instructions
│   ├── shipping_policy.md     # Shipping information
│   ├── return_process.md      # Return and refund policies
│   ├── product_features.md    # Product feature descriptions
│   └── billing_support.md     # Billing and payment info
├── data/
│   └── mock_crm_data.csv     # Simulated customer data
├── langgraph_triage.py       # Main LangGraph orchestration
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
└── .env.example             # Environment variables template
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the environment template and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the System

```bash
python main.py
```

## 💻 Usage

### Interactive CLI Mode

1. Start the system: `python main.py`
2. Enter customer queries when prompted
3. View the classification, response, and escalation decisions
4. Type `quit` to exit or `test` to run predefined test queries

### Example Queries

Try these sample queries to see different system behaviors:

- **Technical**: "I forgot my password, how do I reset it?"
- **Shipping**: "When will my order arrive?"
- **Escalation**: "I want to speak to a manager! This service is terrible!"
- **Product**: "What features are included in the premium plan?"
- **Returns**: "How do I return this item?"

### Test Mode

Enter `test` in the CLI to run a comprehensive set of test queries that demonstrate various classification categories and escalation scenarios.

## 🧠 System Capabilities

### Query Classification
Automatically categorizes queries into:
- `billing` - Payment, subscription, pricing questions
- `technical` - Login, password, app issues
- `shipping` - Delivery, tracking, policies
- `returns` - Returns, refunds, exchanges
- `product` - Features, capabilities, comparisons
- `general` - General inquiries, feedback
- `escalate` - Complex issues requiring human intervention

### Knowledge Base Integration
- Semantic search across documentation
- Context-aware response generation
- Automatic retrieval of relevant policies and procedures

### Customer Data Lookup
- Simulated CRM integration
- Customer account status checking
- Subscription tier information
- Purchase history access

### Intelligent Escalation
Automatically escalates queries based on:
- Emotional sentiment analysis
- Complexity assessment
- Specific keyword detection
- Response confidence levels

## 🔧 Customization

### Adding New Knowledge Base Documents
1. Create new `.md` files in the `knowledge_base/` directory
2. The RAG system will automatically index new content
3. Restart the system to reload the knowledge base

### Modifying Customer Data
Edit `data/mock_crm_data.csv` to add or modify customer records

### Adjusting Classification Categories
Modify the system prompt in `agents/classifier_agent.py` to add new categories or change classification logic

### Customizing Escalation Rules
Update `agents/escalation_agent.py` to modify escalation keywords or logic

## 🎯 Skills Demonstrated

This project showcases key AI Business Analyst capabilities:

1. **Multi-Agent Orchestration**: Complex workflow management with LangGraph
2. **RAG Implementation**: Retrieval-augmented generation for knowledge base queries
3. **AI Agent Design**: Specialized agents with distinct roles and responsibilities
4. **Tool Integration**: Custom tools for data access and processing
5. **Decision Automation**: Intelligent routing and escalation logic
6. **System Architecture**: Modular, scalable design patterns
7. **Error Handling**: Robust fallback mechanisms and error recovery

## 🔍 Technical Details

- **Framework**: LangGraph for multi-agent orchestration
- **LLM**: OpenAI GPT-3.5-turbo for natural language processing
- **Vector Store**: FAISS for document embeddings and similarity search
- **Text Processing**: LangChain for document loading and text splitting
- **Data Handling**: Pandas for CSV data manipulation

## 📈 Future Enhancements

Potential improvements for production deployment:
- Database integration for persistent customer data
- Advanced sentiment analysis
- Multi-language support
- Integration with ticketing systems
- Performance monitoring and analytics
- A/B testing for response optimization

---

Built with ❤️ using LangGraph, LangChain, and OpenAI