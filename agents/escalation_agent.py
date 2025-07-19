from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

class EscalationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.escalation_keywords = [
            "manager", "supervisor", "complaint", "lawsuit", "legal",
            "unacceptable", "terrible", "worst", "angry", "furious",
            "refund immediately", "cancel everything", "speak to someone",
            "this is ridiculous", "demand", "compensation"
        ]
    
    def should_escalate(self, query: str, category: str, response_confidence: float = 0.5) -> dict:
        """Determine if a query should be escalated to human support."""
        
        # Always escalate if classified as 'escalate'
        if category == "escalate":
            return {
                "escalate": True,
                "reason": "Query classified as requiring human intervention",
                "priority": "high"
            }
        
        # Check for escalation keywords
        query_lower = query.lower()
        found_keywords = [kw for kw in self.escalation_keywords if kw in query_lower]
        
        if found_keywords:
            return {
                "escalate": True,
                "reason": f"Escalation keywords detected: {', '.join(found_keywords)}",
                "priority": "high"
            }
        
        # Use LLM to analyze sentiment and complexity
        escalation_check = self._analyze_escalation_need(query, category)
        
        if escalation_check["escalate"]:
            return escalation_check
        
        # Check response confidence
        if response_confidence < 0.3:
            return {
                "escalate": True,
                "reason": "Low confidence in automated response",
                "priority": "medium"
            }
        
        return {
            "escalate": False,
            "reason": "Query can be handled by automated system",
            "priority": "low"
        }
    
    def _analyze_escalation_need(self, query: str, category: str) -> dict:
        """Use LLM to analyze if escalation is needed."""
        try:
            system_prompt = """You are an escalation decision agent. Analyze the customer query and determine if it needs human intervention.

Escalate if the query involves:
- Strong negative emotions or complaints
- Complex technical issues beyond basic troubleshooting
- Requests for compensation or refunds
- Legal or compliance matters
- Threats or aggressive language
- Multiple failed previous attempts
- Account security concerns

Respond with a JSON-like format:
{
    "escalate": true/false,
    "reason": "brief explanation",
    "priority": "low/medium/high"
}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Category: {category}\nQuery: {query}")
            ]
            
            response = self.llm(messages)
            
            # Simple parsing - in production, use proper JSON parsing
            content = response.content.lower()
            if "escalate\": true" in content or "escalate\":true" in content:
                if "priority\": \"high\"" in content:
                    priority = "high"
                elif "priority\": \"medium\"" in content:
                    priority = "medium"
                else:
                    priority = "medium"
                
                return {
                    "escalate": True,
                    "reason": "LLM analysis indicates escalation needed",
                    "priority": priority
                }
            
            return {
                "escalate": False,
                "reason": "LLM analysis indicates no escalation needed",
                "priority": "low"
            }
        
        except Exception as e:
            # If analysis fails, err on the side of caution
            return {
                "escalate": True,
                "reason": f"Error in escalation analysis: {str(e)}",
                "priority": "medium"
            }