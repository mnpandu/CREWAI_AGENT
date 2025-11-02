# load_website_to_pgvector.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from pgvector_setup import vector_store
import os

def load_site(url: str):
    print(f"🔹 Loading: {url}")

    # 1️⃣ Detect PDF or HTML
    try:
        if url.lower().endswith(".pdf"):
            # Use PDF loader for proper page-wise chunking
            pdf_path = "temp.pdf"
            os.system(f"curl -L -o {pdf_path} {url}")  # download the PDF locally

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            os.remove(pdf_path)
        else:
            loader = WebBaseLoader(url)
            docs = loader.load()

        # 2️⃣ Split into smaller text chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        # 3️⃣ Add to pgvector
        if chunks:
            vector_store.add_documents(chunks)
            print(f"✅ Stored {len(chunks)} chunks from {url} into pgvector\n")
        else:
            print(f"⚠️ No text found in {url}")

    except Exception as e:
        print(f"❌ Error loading {url}: {e}")

if __name__ == "__main__":
    sites = [
        "https://www.guidewellsource.com/about.html",
        "https://www.cms.gov/regulations-and-guidance/guidance/manuals/downloads/clm104c03.pdf",
    ]
    for site in sites:
        load_site(site)
