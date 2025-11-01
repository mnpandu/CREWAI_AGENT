from langchain.schema import Document
from pgvector_setup import retriever
from grader import RetrievalGrader
from generator import rag_chain
from rewriter import question_rewriter
from dotenv import load_dotenv

load_dotenv()


def retrieve(state):
    """
    Retrieve documents from pgvector retriever only.
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate final answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Grade retrieved docs for relevance to the query.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    grader = RetrievalGrader()
    filtered_docs = []

    for d in documents:
        score = grader.grade(question, d.page_content)
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Rewrite the query to improve retrieval quality.
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def decide_to_generate(state):
    """
    Always proceed to generation (no web search branch).
    """
    print("---ASSESS GRADED DOCUMENTS---")
    print("---DECISION: GENERATE---")
    return "generate"
