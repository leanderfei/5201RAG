import asyncio
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from retrieve import rephrase_retrieve, get_rag_chain, get_llm, get_retriever

# Store conversation history
chat_history = []

# 1) Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="./bge-base-zh-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    }, # Output normalized vectors for cosine similarity
)

# 2) Initialize LLM
llm = get_llm()

async def invoke_rag(query,conversation_id,chat_history):

    answer = ""

    input={"query":query,"history":chat_history}

    # 1) Get retriever
    retriever=get_retriever(k=20,embedding_model=embedding_model)
    # 2) Rewrite query and retrieve
    retrieve_result= rephrase_retrieve(input,llm,retriever)
    # 3) Build RAG chain
    rag_chain = get_rag_chain(retrieve_result,llm)
    # 4) Execute RAG chain asynchronously with streaming output
    async for chunk in rag_chain.astream(input):
        answer += chunk
        yield chunk # Stream each chunk instead of waiting for the full answer.

    # 5) Update conversation history with user query and assistant answer
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id}
    )
    chat_history.append(
        {"role": "ai", "content": answer, "conversation_id": conversation_id}
    )


if __name__ == '__main__':
    async def main():
        query_list = [
            "How is the SpatialTemporal Co-Attention mechanism constructed?",
            "Which paper uses federated learning, and what benefit does it claim?"
        ]
        for query in query_list:
            print(f"===== Query: {query} =====")
            async for chunk in invoke_rag(query,1,chat_history):
                print(chunk, end="", flush=True)

    asyncio.run(main())