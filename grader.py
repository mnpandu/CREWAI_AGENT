# retrieval_grader_module.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGrader:
    """
    Reusable Retrieval Grader — checks if a retrieved document
    is relevant to the user's question using an LLM binary output.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        # 1️⃣ Create the model
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # 2️⃣ Add structured output schema
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        # 3️⃣ Build the system prompt
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

        # 4️⃣ Combine into a ChatPromptTemplate
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
            ]
        )

        # 5️⃣ Combine into a runnable chain
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

    def grade(self, question: str, document: str) -> GradeDocuments:
        """
        Grade a single document for relevance.
        Returns a structured GradeDocuments object.
        """
        return self.retrieval_grader.invoke({"question": question, "document": document})