from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from tools.rag_tool import RAGTool
from tools.crm_tool import CRMTool
from dotenv import load_dotenv

load_dotenv()

class ResponseGenerationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        self.rag_tool = RAGTool()
        self.crm_tool = CRMTool()
        self.tools = [
            self.rag_tool.get_tool(),
            self.crm_tool.get_account_status_tool()
        ]
        
        self.prompt_template = PromptTemplate.from_template("""
You are a helpful customer support agent. Use the available tools to find relevant information and provide a comprehensive, helpful response to the customer's query.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Guidelines:
1. Always be polite and professional
2. If you find relevant information in the knowledge base, use it to provide a detailed answer
3. If customer information is needed and provided, look it up
4. Be specific and actionable in your responses
5. If you cannot find sufficient information, acknowledge this and suggest contacting human support

Query Category: {category}
Customer Query: {input}
Additional Context: {context}

{agent_scratchpad}""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt_template
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True
        )
    
    def generate_response(self, query: str, category: str, context: str = "") -> str:
        """Generate a response using available tools and context."""
        try:
            result = self.agent_executor.invoke({
                "input": query,
                "category": category,
                "context": context
            })
            return result["output"]
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request. Please contact our human support team for assistance. Error: {str(e)}"
    
    def simple_response(self, query: str, category: str, retrieved_info: str = "") -> str:
        """Generate a simple response without agent executor for fallback."""
        try:
            system_prompt = f"""You are a helpful customer support agent. 
            
The customer's query has been classified as: {category}

{f"Here's relevant information from our knowledge base: {retrieved_info}" if retrieved_info else ""}

Please provide a helpful, professional response to the customer's query. If you don't have enough information, acknowledge this and suggest contacting human support."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm(messages)
            return response.content
        except Exception as e:
            return "I apologize, but I'm unable to process your request at the moment. Please contact our human support team for assistance."