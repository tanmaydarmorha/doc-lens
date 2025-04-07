"""
LLM Service for document analysis project.

This module provides an agentic RAG implementation using LangGraph for document querying.
It follows the LangGraph agentic RAG pattern to provide high-quality responses to user questions.
"""

import uuid
from typing import Any, Dict, List, Optional, Sequence, Annotated, Literal
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from pydantic import BaseModel, Field

from app.utils import get_chat_model, get_embedding_model


class AgentState(TypedDict):
    """
    State object for the LangGraph agent.
    
    The state consists of a sequence of messages that are passed between nodes.
    Each node appends to the messages list using the add_messages annotation.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tracking_id: str


class DocumentGrader(BaseModel):
    """Binary score for document relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class LLMService:
    """
    Service for handling document queries using an agentic RAG approach.
    
    This service implements a LangGraph workflow that:
    1. Takes a user question
    2. Determines if retrieval is needed
    3. Retrieves relevant documents
    4. Grades document relevance
    5. Either generates a response or rewrites the query
    6. Returns the final answer
    """
    
    def __init__(self, vectorstore: Optional[FAISS] = None):
        """
        Initialize the LLM service.
        
        Args:
            vectorstore: Optional FAISS vectorstore for document retrieval
        """
        self.vectorstore = vectorstore
        self.chat_model = get_chat_model(temperature=0, verbose=True)
        self.embedding_model = get_embedding_model()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for agentic RAG.
        
        Returns:
            A compiled StateGraph that can be executed
        """
        # Create retriever tool if vectorstore is available
        tools = []
        if self.vectorstore is not None:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            retriever_tool = create_retriever_tool(
                retriever,
                "search_documents",
                "Search for information in the document database. Use this for specific information needs.",
            )
            tools.append(retriever_tool)
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        
        if tools:
            retrieve = ToolNode(tools)
            workflow.add_node("retrieve", retrieve)
            workflow.add_node("grade_documents", self._grade_documents_node)
        
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("rewrite", self._rewrite_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        
        if tools:
            # Decide whether to retrieve
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "retrieve",
                    END: END,
                },
            )
            
            # After retrieval, grade documents
            workflow.add_conditional_edges(
                "retrieve",
                self._grade_documents,
                {
                    "generate": "generate",
                    "rewrite": "rewrite",
                },
            )
        else:
            # If no tools, go straight to generate
            workflow.add_edge("agent", "generate")
        
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        # Compile
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> Dict:
        """
        Agent node that decides whether to use tools or generate a response.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Updated state with agent response
        """
        messages = state["messages"]
        model = self.chat_model
        
        if hasattr(self, "vectorstore") and self.vectorstore is not None:
            # Bind tools if available
            tools = []
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            retriever_tool = create_retriever_tool(
                retriever,
                "search_documents",
                "Search for information in the document database. Use this for specific information needs.",
            )
            tools.append(retriever_tool)
            model = model.bind_tools(tools)
        
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def _grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
        """
        Determines whether retrieved documents are relevant to the question.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Decision string: "generate" if documents are relevant, "rewrite" if not
        """
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        # Create a structured output model for grading
        model = self.chat_model.with_structured_output(DocumentGrader)
        
        # Prompt for document grading
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.
            
            Here is the retrieved document: 
            
            {context}
            
            Here is the user question: {question}
            
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        
        # Chain
        chain = prompt | model
        
        # Invoke the chain
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        
        if score == "yes":
            return "generate"
        else:
            return "rewrite"
    
    def _grade_documents_node(self, state: AgentState) -> Dict:
        """
        Node wrapper for the grade_documents function.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Updated state with grading result
        """
        result = self._grade_documents(state)
        return {"messages": [AIMessage(content=f"Document relevance: {result}")]}
    
    def _rewrite_node(self, state: AgentState) -> Dict:
        """
        Transform the query to produce a better question.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content
        
        msg = [
            HumanMessage(
                content=f"""
                Look at the input and try to reason about the underlying semantic intent / meaning.
                
                Here is the initial question:
                ------- 
                {question} 
                ------- 
                
                Formulate an improved question: """
            )
        ]
        
        # Use chat model to rewrite the question
        model = self.chat_model
        response = model.invoke(msg)
        
        # Return the rewritten question as a new human message
        return {"messages": [HumanMessage(content=response.content)]}
    
    def _generate_node(self, state: AgentState) -> Dict:
        """
        Generate an answer based on retrieved documents.
        
        Args:
            state: Current agent state with messages
            
        Returns:
            Updated state with generated answer
        """
        messages = state["messages"]
        question = messages[0].content
        
        # Get the most recent retrieval result if available
        docs_content = ""
        for msg in reversed(messages):
            if msg.content and isinstance(msg.content, str) and len(msg.content) > 100:
                docs_content = msg.content
                break
        
        # RAG prompt
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant for answering questions based on provided documents.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        INSTRUCTIONS:
        - Answer the question based only on the provided context
        - If the context doesn't contain the answer, say "I don't have enough information to answer this question"
        - Provide a detailed and helpful response
        - Format your response in a clear and readable way
        """)
        
        # Chain
        rag_chain = prompt | self.chat_model | StrOutputParser()
        
        # Generate response
        response = rag_chain.invoke({
            "context": docs_content,
            "question": question
        })
        
        return {"messages": [AIMessage(content=response)]}
    
    def query(self, question: str, tracking_id: Optional[str] = None) -> str:
        """
        Process a user query through the agentic RAG workflow.
        
        Args:
            question: The user's question
            tracking_id: Optional tracking ID for the conversation
            
        Returns:
            The generated response
        """
        if tracking_id is None:
            tracking_id = str(uuid.uuid4())
        
        # Initialize state with the user's question
        inputs = {
            "messages": [HumanMessage(content=question)],
            "tracking_id": tracking_id
        }
        
        # Execute the graph
        outputs = self.graph.invoke(inputs)
        
        # Extract the final response from the messages
        messages = outputs.get("messages", [])
        if messages:
            # Get the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
        
        # Fallback response
        return "I couldn't generate a response. Please try again."