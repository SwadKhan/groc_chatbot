from basic_rag import BasicRAG
import os

print("\n" + "=" * 80)
print("SETTING UP VECTOR STORE FOR RAG PIPELINE")
print("=" * 80)


rag = BasicRAG(chunk_size=1000, chunk_overlap=200, query_limit=50, max_tokens=300)

# Check for documents
if not os.path.exists("documents"):
    print("❌ Error: 'documents' folder not found!")
    print("   Please create it and add your PDF files.")
    exit(1)

pdf_files = [f for f in os.listdir("documents") if f.endswith(".pdf")]

if not pdf_files:
    print("❌ Error: No PDF files found in 'documents' folder!")
    print("   Please add your PDF files to the 'documents' folder.")
    exit(1)

print(f"\n✅ Found {len(pdf_files)} PDF file(s):")
for pdf in pdf_files:
    print(f"   • {pdf}")

pdf_paths = [os.path.join("documents", pdf) for pdf in pdf_files]

print("\n🔄 Initializing RAG system...")
rag = BasicRAG(chunk_size=1000, chunk_overlap=200)

print("\n📚 Loading documents...")
rag.load_documents(pdf_paths, doc_type="pdf")

print("\n🔮 Creating vector embeddings...")
rag.create_vectorstore("./chroma_db")

print("\n⚙️ Setting up QA chain...")
rag.setup_chain(k=3, temperature=0.0)

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE!")
print("=" * 80)
print("\nYou can now run: streamlit run streamlit_app.py")
print("The app will automatically load this knowledge base!")
print("=" * 80)
