# create_database.py
# Run this ONCE to create the database from 11000 PDFs
# After this you never need to run it again

import os
import glob
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import *

def create_database():
    print("=" * 60)
    print("   CREATING DATABASE FROM ALL 11000 PDFs")
    print("=" * 60)
    print("This will take 2-4 hours for 11000 files")
    print("Keep your laptop plugged in!\n")

    # ----------------------------------------
    # STEP 1: Collect all PDF paths
    # ----------------------------------------
    all_pdfs = []
    for folder in PDF_FOLDERS:
        if os.path.exists(folder):
            pdfs = glob.glob(f"{folder}/*.pdf")
            print(f"📁 {os.path.basename(folder)}: {len(pdfs)} PDFs")
            all_pdfs.extend(pdfs)
        else:
            print(f"❌ Folder not found: {folder}")

    print(f"\n✅ Total PDFs found: {len(all_pdfs)}")

    if len(all_pdfs) == 0:
        print("No PDFs found! Check your folder paths in config.py")
        return

    # ----------------------------------------
    # STEP 2: Load PDFs
    # ----------------------------------------
    print("\nLoading PDFs...")
    docs = []
    failed = 0
    start_time = time.time()

    for i, pdf in enumerate(all_pdfs):
        try:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            docs.extend(pages)
        except:
            failed += 1

        # Progress update every 500 files
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (i+1)) * (len(all_pdfs) - i - 1)
            print(f"📄 {i+1}/{len(all_pdfs)} files | "
                  f"Pages: {len(docs)} | "
                  f"Time left: {remaining/60:.1f} mins")

    print(f"\n✅ Loading done!")
    print(f"Total pages: {len(docs)}")
    print(f"Failed files: {failed}")

    # ----------------------------------------
    # STEP 3: Split into chunks
    # ----------------------------------------
    print("\nSplitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(docs)
    print(f"✅ Total chunks: {len(splits)}")

    # ----------------------------------------
    # STEP 4: Load embedding model
    # ----------------------------------------
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    print("✅ Embedding model ready!")

    # ----------------------------------------
    # STEP 5: Create FAISS database in batches
    # ----------------------------------------
    print("\nCreating vector database...")
    print("This is the longest step - please wait...")

    os.makedirs(FAISS_PATH, exist_ok=True)
    batch_size = 5000
    vectorstore = None

    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        # Save after every batch (safety)
        vectorstore.save_local(FAISS_PATH)

        print(f"✅ Batch {i//batch_size + 1} done | "
              f"Chunks processed: {min(i+batch_size, len(splits))}/{len(splits)} | "
              f"Vectors: {vectorstore.index.ntotal}")

    print(f"\n✅ DATABASE CREATED SUCCESSFULLY!")
    print(f"Total vectors stored: {vectorstore.index.ntotal}")
    print(f"Database saved at: {FAISS_PATH}")
    print("\nNow run: python app.py")

if __name__ == "__main__":
    create_database()