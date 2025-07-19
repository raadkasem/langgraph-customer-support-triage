import os
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv()

class RAGTool:
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.vector_store = None
        self.qa_chain = None
        self._setup_rag()
    
    def _setup_rag(self):
        """Load documents, create embeddings, and set up RAG chain."""
        loader = DirectoryLoader(self.knowledge_base_path, glob="*.md")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )
    
    def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for relevant information."""
        try:
            result = self.qa_chain.run(query)
            return result
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    def get_tool(self) -> Tool:
        """Return the RAG tool for use by agents."""
        return Tool(
            name="knowledge_base_search",
            description="Search the knowledge base for information about password reset, shipping, returns, billing, and product features. Use this when customers ask questions that might be answered in our documentation.",
            func=self.search_knowledge_base
        )