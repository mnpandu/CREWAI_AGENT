from langchain_core.documents import Document
from pgvector_setup import retriever
from grader import RetrievalGrader
from generator import rag_chain,format_docs
from rewriter import question_rewriter
from langchain_tavily import TavilySearch
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
    # If at least one relevant document, web search is "Yes"
    web_search = "No" if len(filtered_docs) > 0 else "Yes"
    print(f"---WEB SEARCH NEEDED: {web_search}---")
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


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
    print(f"---NEW QUESTION: {better_question}---")
    return {"documents": documents, "question": better_question, "timesTransformed": timesTransformed}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

from datetime import datetime
def web_search(state):
    """
    Perform web search via Tavily if no relevant documents found in vector DB.
    Adds metadata before storing in the retriever.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Initialize TavilySearch
    try:
        search_tool = TavilySearch(max_results=3)
        results = search_tool.invoke(question)
    except Exception as e:
        print(f"❌ Web search failed: {e}")
        return {"documents": documents, "question": question}

    # Normalize results to Document objects with metadata
    new_docs = []
    if isinstance(results, list):
        for r in results:
            if isinstance(r, Document):
                # Add metadata if not present
                r.metadata.update({
                    "source": r.metadata.get("source", "TavilySearch"),
                    "search_query": question,
                    "retrieved_from": "web_search",
                    "timestamp": datetime.now().isoformat()
                })
                new_docs.append(r)
            else:
                new_docs.append(
                    Document(
                        page_content=str(r),
                        metadata={
                            "source": "TavilySearch",
                            "search_query": question,
                            "retrieved_from": "web_search",
                            "timestamp": datetime.now().isoformat()
                        },
                    )
                )
    else:
        new_docs.append(
            Document(
                page_content=str(results),
                metadata={
                    "source": "TavilySearch",
                    "search_query": question,
                    "retrieved_from": "web_search",
                    "timestamp": datetime.now().isoformat()
                },
            )
        )

    # Add results to memory store
    retriever.add_documents(new_docs)
    documents.extend(new_docs)

    print(f"---WEB SEARCH COMPLETE: {len(new_docs)} new docs stored---")
    return {"documents": documents, "question": question}
