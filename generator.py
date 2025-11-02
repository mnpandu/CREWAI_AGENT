# generator.py
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langsmith import Client
hub= Client()
# Load prompt template
prompt = hub.pull_prompt("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Chain definition: prompt → llm → string output
rag_chain = prompt | llm | StrOutputParser()

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)