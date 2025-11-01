# generator.py
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load prompt template
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Chain definition: prompt → llm → string output
rag_chain = prompt | llm | StrOutputParser()
