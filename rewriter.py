# question_rewriter.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the LLM (you can switch to gpt-4o for better performance)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# System prompt for rewriting user questions
system = (
    "You are a question re-writer that converts an input question into "
    "a clearer, search-optimized version. Focus on improving clarity and "
    "capturing the underlying semantic intent or meaning."
)

# Create the chat prompt
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question:\n\n{question}\n\n"
            "Formulate an improved version suitable for search or retrieval.",
        ),
    ]
)

# Build the LangChain runnable pipeline
question_rewriter = re_write_prompt | llm | StrOutputParser()
