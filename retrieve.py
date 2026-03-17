import os
from typing import Dict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def format_history(history,max_epoch=3):
    # Each turn includes a user query and an assistant response.
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch :]
    return "\n".join([f"{i['role']}: {i['content']}" for i in history])

def format_docs(docs: list[Document]) -> str:
    """Concatenate page_content from multiple Documents."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(k=20,embedding_model=None):
    """Get a retriever from the vector database."""
    # 1) Initialize Chroma client
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embedding_model,
    )

    # 2) Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity", # Retrieval mode: similarity or mmr
        search_kwargs={"k": k},
    )
    
    return retriever


def get_llm():
    # LLM
    load_dotenv()
    TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
    llm = Tongyi(model="qwen-turbo", api_key=TONGYI_API_KEY)
    return llm

def rephrase_retrieve(input:Dict[str,str],llm,retriever):
    """Rewrite user query and retrieve from vector database."""
    
    # 1) Prompt for query rewriting
    rephrase_prompt = PromptTemplate.from_template(
    """
    Improve the latest user message based on conversation history so it becomes more specific.
    Output only the improved question.
    If no improvement is needed, output the original question directly.
    
    {history}
    User: {query}
    """
    )
    
    # 2) Rephrase chain: generate a more specific query from history + current query
    rephrase_chain = (
        {
            "history": lambda x :format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | rephrase_prompt
        | llm
        | StrOutputParser()
        | (lambda x: print(f"===== Rewritten Query: {x} =====") or x)
    )
    
    # 3) Execute rewrite
    rephrase_query = rephrase_chain.invoke({"history": input.get("history"), "query": input.get("query")})

    # 4) Retrieve with rewritten query
    retrieve_result = retriever.invoke(rephrase_query,k=3)

    return retrieve_result

def get_rag_chain(retrieve_result,llm):
    """Build RAG chain using retrieved context, history, and user query."""

    # 1) Prompt template
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""
    You are a professional academic QA assistant.
    Answer only based on the provided context and conversation history.
    If the answer is not available, reply: "I don't know".

    Context: {context}

    History: [{history}]

    Question: {query}

    Answer:""",
    )

    # 2) Define RAG chain
    rag_chain = (
        {
            "context": lambda x:format_docs(retrieve_result),
            "history": lambda x: format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | prompt
        | (lambda x: print(x.text, end="") or x) # Print prompt for debugging
        | llm
        | StrOutputParser() # Parse output into plain string
    )

    return rag_chain