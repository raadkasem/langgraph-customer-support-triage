from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agents.classifier_agent import QueryClassifierAgent
from agents.response_agent import ResponseGenerationAgent
from agents.escalation_agent import EscalationAgent
from tools.rag_tool import RAGTool
import json

class TriageState(TypedDict):
    query: str
    classification: str
    retrieved_info: str
    response: str
    escalation_decision: dict
    final_output: str

class CustomerSupportTriageSystem:
    def __init__(self):
        self.classifier = QueryClassifierAgent()
        self.response_agent = ResponseGenerationAgent()
        self.escalation_agent = EscalationAgent()
        self.rag_tool = RAGTool()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(TriageState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("search_knowledge", self._search_knowledge_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("check_escalation", self._check_escalation_node)
        workflow.add_node("escalate", self._escalate_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry point
        workflow.set_entry_point("classify")
        
        # Add edges
        workflow.add_edge("classify", "search_knowledge")
        workflow.add_edge("search_knowledge", "generate_response")
        workflow.add_edge("generate_response", "check_escalation")
        
        # Conditional routing after escalation check
        workflow.add_conditional_edges(
            "check_escalation",
            self._should_escalate,
            {
                "escalate": "escalate",
                "continue": "finalize"
            }
        )
        
        workflow.add_edge("escalate", END)
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _classify_node(self, state: TriageState) -> TriageState:
        """Classify the customer query."""
        classification = self.classifier.classify_query(state["query"])
        return {**state, "classification": classification}
    
    def _search_knowledge_node(self, state: TriageState) -> TriageState:
        """Search the knowledge base for relevant information."""
        if state["classification"] in ["technical", "shipping", "returns", "billing", "product"]:
            retrieved_info = self.rag_tool.search_knowledge_base(state["query"])
        else:
            retrieved_info = ""
        return {**state, "retrieved_info": retrieved_info}
    
    def _generate_response_node(self, state: TriageState) -> TriageState:
        """Generate a response based on the query and retrieved information."""
        try:
            # Try using the agent with tools first
            response = self.response_agent.generate_response(
                state["query"], 
                state["classification"]
            )
        except Exception:
            # Fallback to simple response
            response = self.response_agent.simple_response(
                state["query"], 
                state["classification"], 
                state.get("retrieved_info", "")
            )
        
        return {**state, "response": response}
    
    def _check_escalation_node(self, state: TriageState) -> TriageState:
        """Check if the query should be escalated."""
        escalation_decision = self.escalation_agent.should_escalate(
            state["query"], 
            state["classification"]
        )
        return {**state, "escalation_decision": escalation_decision}
    
    def _escalate_node(self, state: TriageState) -> TriageState:
        """Handle escalation to human support."""
        escalation_msg = f"""
🚨 ESCALATION REQUIRED 🚨

Priority: {state['escalation_decision']['priority'].upper()}
Reason: {state['escalation_decision']['reason']}

Customer Query: {state['query']}
Classification: {state['classification']}

This query has been flagged for human intervention. A human support agent should handle this request.
"""
        return {**state, "final_output": escalation_msg}
    
    def _finalize_node(self, state: TriageState) -> TriageState:
        """Finalize the response for automated handling."""
        final_output = f"""
✅ AUTOMATED RESPONSE

Classification: {state['classification']}

Response: {state['response']}

---
This response was generated automatically. If you need further assistance, please contact our human support team.
"""
        return {**state, "final_output": final_output}
    
    def _should_escalate(self, state: TriageState) -> str:
        """Determine routing based on escalation decision."""
        return "escalate" if state["escalation_decision"]["escalate"] else "continue"
    
    def process_query(self, query: str) -> dict:
        """Process a customer query through the entire triage system."""
        initial_state = {
            "query": query,
            "classification": "",
            "retrieved_info": "",
            "response": "",
            "escalation_decision": {},
            "final_output": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "classification": result["classification"],
            "escalated": result["escalation_decision"].get("escalate", False),
            "priority": result["escalation_decision"].get("priority", "low"),
            "output": result["final_output"]
        }

# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in a .env file")
        exit(1)
    
    system = CustomerSupportTriageSystem()
    
    # Test queries
    test_queries = [
        "I forgot my password, how do I reset it?",
        "When will my order arrive?",
        "I want to speak to a manager immediately! This service is terrible!",
        "What features are included in the premium plan?",
        "I want to return this item, what's the process?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = system.process_query(query)
        print(result["output"])