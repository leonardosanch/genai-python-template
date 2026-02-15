"""
Complete RAG Pipeline Example

This example demonstrates:
- Document loading and chunking
- Vector store integration (Pinecone)
- LLM integration (OpenAI)
- Evaluation with RAGAS
"""

import asyncio
import os

import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness


async def setup_vector_store(documents_path: str) -> Pinecone:
    """Load documents and create vector store."""
    # Load documents
    loader = TextLoader(documents_path)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="rag-example")

    return vectorstore


async def create_rag_chain(vectorstore: Pinecone) -> RetrievalQA:
    """Create RAG chain with retriever and LLM."""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


async def evaluate_rag(qa_chain: RetrievalQA, test_questions: list[dict]):
    """Evaluate RAG pipeline with RAGAS."""
    results = []

    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        response = await qa_chain.ainvoke({"query": question})

        results.append(
            {
                "question": question,
                "answer": response["result"],
                "contexts": [doc.page_content for doc in response["source_documents"]],
                "ground_truth": ground_truth,
            }
        )

    # Evaluate with RAGAS
    evaluation = evaluate(
        results,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n=== RAG Evaluation Results ===")
    print(f"Faithfulness: {evaluation['faithfulness']:.3f}")
    print(f"Answer Relevancy: {evaluation['answer_relevancy']:.3f}")
    print(f"Context Precision: {evaluation['context_precision']:.3f}")

    return evaluation


async def main():
    """Run RAG pipeline example."""
    # Setup
    print("Setting up vector store...")
    vectorstore = await setup_vector_store("data/documents.txt")

    print("Creating RAG chain...")
    qa_chain = await create_rag_chain(vectorstore)

    # Test questions
    test_questions = [
        {
            "question": "What is Clean Architecture?",
            "ground_truth": (
                "Clean Architecture is a software design philosophy that separates "
                "concerns into layers."
            ),
        },
        {
            "question": "What are the SOLID principles?",
            "ground_truth": (
                "SOLID is an acronym for five design principles: SRP, OCP, LSP, ISP, and DIP."
            ),
        },
    ]

    # Query
    print("\n=== Running Queries ===")
    for item in test_questions:
        response = await qa_chain.ainvoke({"query": item["question"]})
        print(f"\nQ: {item['question']}")
        print(f"A: {response['result']}")
        print(f"Sources: {len(response['source_documents'])}")

    # Evaluate
    print("\n=== Evaluating RAG Pipeline ===")
    await evaluate_rag(qa_chain, test_questions)


if __name__ == "__main__":
    asyncio.run(main())
