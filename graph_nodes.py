from langchain.schema import Document
from pgvector_setup import retriever
from grader import RetrievalGrader
from generator import rag_chain,format_docs
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
    return {"documents": documents, "question": question, "timesTransformed": 0}


def generate(state):
    """
    Generate final answer using RAG.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
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
        print(d.metadata['source'],f'---SCORE: {score.binary_score}---')
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
    timesTransformed = state["timesTransformed"]
    timesTransformed += 1

    better_question = question_rewriter.invoke({"question": question})
    print("---NEW QUESTION---")
    return {"documents": documents, "question": better_question, "timesTransformed": timesTransformed}


def decide_to_generate(state):
    """
    Always proceed to generation (no web search branch).
    """
    print("---ASSESS GRADED DOCUMENTS---")
    print("---DECISION: GENERATE---")
    return "generate"
