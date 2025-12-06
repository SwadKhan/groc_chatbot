"""
Basic RAG Pipeline Implementation (Groq LLM + FastEmbed Embeddings)
with cost-control strategies:
- Cheap Groq model (llama-3.1-8b-instant by default)
- Local embeddings (FastEmbed)
- max_tokens limit
- In-memory caching of answers
- Simple usage tracking + per-session query limit
"""

import os
import time
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq  # Groq LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load GROQ_API_KEY from .env
load_dotenv()


class BasicRAG:
    """
    Basic RAG Pipeline with:
      - local FastEmbedEmbeddings
      - Groq LLM (llama-3.1-8b-instant by default)
      - in-memory caching
      - simple usage tracking
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        query_limit: Optional[int] = 100,
        max_tokens: int = 300,
        groq_model: str = "llama-3.1-8b-instant",
    ):
        """
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            query_limit: Max number of paid LLM queries per session (None = no cap)
            max_tokens: Max tokens per LLM response (cost control)
            groq_model: Groq model name (e.g. 'llama-3.1-8b-instant')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.vectorstore = None
        self.qa_chain = None

        # Cost-control configuration
        self.query_limit = query_limit
        self.max_tokens = max_tokens
        self.groq_model = groq_model

        # In-memory cache: key = (question, k, temperature) -> response dict
        self._cache: Dict[Tuple[str, int, float], Dict] = {}

        # Simple usage tracking
        self.total_queries = 0  # number of TIMES we called the LLM via RAG

    # ------------------------------------------------------------------
    # Document loading & chunking
    # ------------------------------------------------------------------

    def load_documents(self, paths: List[str], doc_type: str = "pdf") -> None:
        """Load documents from various sources."""
        all_documents = []

        for path in paths:
            print(f"Loading document: {path}")

            try:
                if doc_type == "pdf":
                    loader = PyPDFLoader(path)
                elif doc_type == "web":
                    loader = WebBaseLoader(path)
                elif doc_type == "txt":
                    loader = TextLoader(path)
                else:
                    raise ValueError(f"Unsupported document type: {doc_type}")

                documents = loader.load()
                all_documents.extend(documents)
                print(f"✓ Loaded {len(documents)} pages/sections from {path}")

            except Exception as e:
                print(f"✗ Error loading {path}: {str(e)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        self.chunks = text_splitter.split_documents(all_documents)
        print(f"\n✓ Total chunks created: {len(self.chunks)}")

    # ------------------------------------------------------------------
    # Vector store (FastEmbed + Chroma)
    # ------------------------------------------------------------------

    def create_vectorstore(self, persist_directory: str = "./chroma_db") -> None:
        """Create vector store from chunks using FastEmbed."""
        if not self.chunks:
            raise ValueError("No documents loaded. Call load_documents() first.")

        print("\nCreating vector embeddings with FastEmbed...")
        embeddings = FastEmbedEmbeddings()

        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        print(f"✓ Vector store created with {len(self.chunks)} chunks")
        print(f"✓ Database persisted at: {persist_directory}")

    def load_vectorstore(self, persist_directory: str = "./chroma_db") -> None:
        """Load existing FastEmbed-based vector store."""
        print("\nLoading existing vector store with FastEmbed...")
        embeddings = FastEmbedEmbeddings()

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        print("✓ Vector store loaded successfully")

    # ------------------------------------------------------------------
    # LLM + Retrieval chain
    # ------------------------------------------------------------------

    def _build_llm(self, temperature: float) -> ChatGroq:
        """
        Create a cost-optimized Groq chat model.
        GROQ_API_KEY is read automatically from the environment.
        """
        return ChatGroq(
            model=self.groq_model,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )

    def setup_chain(self, k: int = 3, temperature: float = 0.0) -> None:
        """Setup RetrievalQA chain."""
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Call create_vectorstore() or load_vectorstore() first."
            )

        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know; don't make anything up.
Always base your answer strictly on the provided context.

Context:
{context}

Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        llm = self._build_llm(temperature=temperature)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        # store config for caching key
        self._current_k = k
        self._current_temperature = temperature

        print(f"\n✓ QA Chain setup complete (retrieving top {k} documents)")

    # ------------------------------------------------------------------
    # Usage tracking helpers
    # ------------------------------------------------------------------

    def get_usage_stats(self) -> Dict:
        """Return simple usage statistics."""
        return {
            "total_queries": self.total_queries,
            "limit": self.query_limit,
            "remaining": None
            if self.query_limit is None
            else max(self.query_limit - self.total_queries, 0),
        }

    def _check_limit(self) -> None:
        """Raise if query limit exceeded."""
        if self.query_limit is not None and self.total_queries >= self.query_limit:
            raise RuntimeError(
                f"Query limit reached ({self.query_limit} calls). "
                "To avoid unexpected Groq costs, further queries are blocked. "
                "You can raise 'query_limit' in BasicRAG(...) if you understand the cost."
            )

    # ------------------------------------------------------------------
    # Query + caching
    # ------------------------------------------------------------------

    def query(self, question: str, verbose: bool = True) -> Dict:
        """
        Query the RAG system with caching + usage tracking.

        Returns:
            Dict with keys: 'result' and 'source_documents'.
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_chain() first.")

        cache_key = (
            question,
            getattr(self, "_current_k", 3),
            getattr(self, "_current_temperature", 0.0),
        )

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if verbose:
                print(f"\n[Cache hit] Returning cached answer for question:\n{question}")
            return cached

        # Enforce query limit
        self._check_limit()

        print(f"\n{'=' * 80}")
        print(f"QUERY: {question}")
        print(f"{'=' * 80}")

        start_time = time.time()
        response = self.qa_chain({"query": question})
        elapsed_time = time.time() - start_time

        # Update usage counter
        self.total_queries += 1

        # Store in cache
        self._cache[cache_key] = response

        if verbose:
            print(f"\nANSWER:\n{response['result']}")
            print(f"\n{'=' * 80}")
            print(f"SOURCE DOCUMENTS ({len(response['source_documents'])}):")
            print(f"{'=' * 80}")

            for i, doc in enumerate(response["source_documents"], 1):
                print(f"\n[Source {i}]")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")

            usage = self.get_usage_stats()
            print(f"\n⏱️  Response time: {elapsed_time:.2f} seconds")
            print(
                f"💰  Total paid queries this session: {usage['total_queries']}"
                + (
                    f" / {usage['limit']} (remaining: {usage['remaining']})"
                    if usage["limit"] is not None
                    else ""
                )
            )

        return response

    # ------------------------------------------------------------------
    # Similarity search (no LLM cost)
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 3) -> List:
        """Perform similarity search using Chroma only (no LLM call => free)."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")

        results = self.vectorstore.similarity_search(query, k=k)

        print(f"\n{'=' * 80}")
        print(f"SIMILARITY SEARCH: {query}")
        print(f"{'=' * 80}")

        for i, doc in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Content: {doc.page_content[:300]}...")
            print(f"Metadata: {doc.metadata}")

        return results
