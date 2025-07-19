from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class QueryClassifierAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.system_prompt = """You are a customer support query classifier. Your job is to classify incoming customer queries into one of these categories:

1. billing - Questions about payments, invoices, subscription changes, pricing
2. technical - Issues with login, password reset, app problems, bugs
3. shipping - Questions about delivery, tracking, shipping costs, policies
4. returns - Questions about returns, refunds, exchanges
5. product - Questions about features, capabilities, comparisons
6. general - General inquiries, compliments, feedback
7. escalate - Complex issues requiring human intervention, complaints, legal matters

Respond with only the category name (lowercase). If unsure between categories, choose the most likely one.

Examples:
- "I forgot my password" → technical
- "When will my order arrive?" → shipping
- "I want to cancel my subscription" → billing
- "How do I return this item?" → returns
- "What features are included in the premium plan?" → product
- "I'm very unhappy with your service and want to speak to a manager" → escalate"""

    def classify_query(self, query: str) -> str:
        """Classify a customer query into predefined categories."""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"Classify this query: {query}")
            ]
            
            response = self.llm.invoke(messages)
            classification = response.content.strip().lower()
            
            valid_categories = ["billing", "technical", "shipping", "returns", "product", "general", "escalate"]
            if classification not in valid_categories:
                return "general"
            
            return classification
        except Exception as e:
            return "general"