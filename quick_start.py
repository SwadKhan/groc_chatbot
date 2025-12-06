"""
Quick Start Interactive Script for RAG Pipeline
Complete code for quick_start.py
Run this to get started quickly!
"""

import os
from dotenv import load_dotenv
from basic_rag import BasicRAG
from advanced_rag import AdvancedRAG

def check_setup():
    """Check if environment is properly set up"""
    print("🔍 Checking setup...")
    
    # Check .env file
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file!")
        return False
    
    print("✅ Environment configured correctly")
    
    # Check documents folder
    if not os.path.exists("documents"):
        print("📁 Creating documents/ folder...")
        os.makedirs("documents")
        print("   Please add your PDF files to the documents/ folder")
        return False
    
    # Check if documents is actually a directory
    if not os.path.isdir("documents"):
        print("❌ 'documents' exists but is not a folder!")
        print("   Please delete the 'documents' file and create a folder instead:")
        print("   Run in PowerShell: Remove-Item documents -Force; New-Item -Path 'documents' -ItemType Directory")
        return False
    
    # Check for documents
    try:
        pdf_files = [f for f in os.listdir("documents") if f.endswith('.pdf')]
    except Exception as e:
        print(f"❌ Error reading documents folder: {e}")
        return False
    
    if not pdf_files:
        print("📄 No PDF files found in documents/ folder")
        print("   Please add at least one PDF file")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s)")
    for f in pdf_files:
        print(f"   - {f}")
    
    return True

def create_new_vectorstore():
    """Create a new vector store from documents"""
    print("\n" + "="*80)
    print("CREATING NEW VECTOR STORE")
    print("="*80)
    
    # Get PDF files
    pdf_files = [
        os.path.join("documents", f) 
        for f in os.listdir("documents") 
        if f.endswith('.pdf')
    ]
    
    print(f"\nFound {len(pdf_files)} document(s) to process")
    
    # Configuration
    chunk_size = input("\nChunk size (default 1000): ").strip()
    chunk_size = int(chunk_size) if chunk_size else 1000
    
    chunk_overlap = input("Chunk overlap (default 200): ").strip()
    chunk_overlap = int(chunk_overlap) if chunk_overlap else 200
    
    print(f"\n📊 Configuration:")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Chunk overlap: {chunk_overlap}")
    print(f"   Documents: {len(pdf_files)}")
    
    input("\nPress Enter to start processing...")
    
    # Create RAG system
    rag = BasicRAG(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Load documents
    rag.load_documents(pdf_files, doc_type="pdf")
    
    # Create vector store
    rag.create_vectorstore("./chroma_db")
    
    # Setup chain
    rag.setup_chain(k=3, temperature=0)
    
    print("\n✅ Vector store created successfully!")
    return rag

def load_existing_vectorstore(use_advanced=False):
    """Load existing vector store"""
    print("\n" + "="*80)
    print("LOADING EXISTING VECTOR STORE")
    print("="*80)
    
    if not os.path.exists("./chroma_db"):
        print("\n❌ No vector store found at ./chroma_db")
        print("   Please create one first using option 1")
        return None
    
    if use_advanced:
        rag = AdvancedRAG()
    else:
        rag = BasicRAG()
    
    rag.load_vectorstore("./chroma_db")
    rag.setup_chain(k=3, temperature=0)
    
    print("\n✅ Vector store loaded successfully!")
    return rag

def interactive_query_loop(rag, use_advanced=False):
    """Interactive query loop"""
    print("\n" + "="*80)
    print("INTERACTIVE QUERY MODE")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question to query the system")
    print("  - Type 'search <query>' for similarity search only")
    print("  - Type 'config' to change settings")
    print("  - Type 'quit' to exit")
    
    if use_advanced:
        print("  - Type 'strategy <n>' to change strategy")
        print("    (strategies: multi_query, hyde, query_rewrite)")
    
    strategy = "basic"
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\n💬 You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'config':
            k = input("Number of documents to retrieve (default 3): ").strip()
            k = int(k) if k else 3
            temp = input("Temperature 0-1 (default 0): ").strip()
            temp = float(temp) if temp else 0
            rag.setup_chain(k=k, temperature=temp)
            print(f"✅ Settings updated (k={k}, temp={temp})")
            continue
        
        if user_input.lower().startswith('search '):
            query = user_input[7:]
            rag.similarity_search(query, k=3)
            continue
        
        if use_advanced and user_input.lower().startswith('strategy '):
            strategy = user_input[9:].strip()
            print(f"✅ Strategy changed to: {strategy}")
            continue
        
        # Process query
        try:
            if use_advanced and isinstance(rag, AdvancedRAG):
                response = rag.advanced_query(
                    user_input,
                    strategy=strategy,
                    k=3,
                    rerank=True
                )
            else:
                response = rag.query(user_input, verbose=True)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def main():
    """Main function"""
    print("="*80)
    print("RAG PIPELINE QUICK START")
    print("="*80)
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please fix the issues above and try again.")
        return
    
    # Main menu
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\n1. Create new vector store from documents")
        print("2. Load existing vector store (Basic RAG)")
        print("3. Load existing vector store (Advanced RAG)")
        print("4. Run test queries")
        print("5. Launch web interface")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            rag = create_new_vectorstore()
            if rag:
                input("\nPress Enter to start querying...")
                interactive_query_loop(rag, use_advanced=False)
        
        elif choice == '2':
            rag = load_existing_vectorstore(use_advanced=False)
            if rag:
                interactive_query_loop(rag, use_advanced=False)
        
        elif choice == '3':
            rag = load_existing_vectorstore(use_advanced=True)
            if rag:
                interactive_query_loop(rag, use_advanced=True)
        
        elif choice == '4':
            print("\n🧪 Running test queries...")
            from test_queries import demo_evaluation
            try:
                demo_evaluation()
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == '5':
            print("\n🌐 Launching web interface...")
            print("   Run: streamlit run streamlit_app.py")
            print("   (Open a new terminal and run the command above)")
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main()