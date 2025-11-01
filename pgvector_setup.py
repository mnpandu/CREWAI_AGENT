# pgvector_setup.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# Load .env
load_dotenv()

# ✅ Create embedding model
embeddings = OpenAIEmbeddings()
db_url=os.getenv("db_url")

# ✅ Create PGVector store (corrected parameters)
vector_store = PGVector(
    connection=db_url,
    embeddings=embeddings,            # <-- changed back to 'embeddings' (plural)
    collection_name="crag_docs",
    use_jsonb=True, # <-- recommended for better performance
)

# ✅ Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})


# Optional test data loader
def add_sample_docs():
    docs = [
        Document(page_content="what is apple"),        
    ]
    vector_store.add_documents(docs)
    print("✅ Sample documents added to PostgreSQL via pgvector.")


if __name__ == "__main__":
    add_sample_docs()