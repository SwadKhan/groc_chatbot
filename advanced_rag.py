"""
Advanced RAG Pipeline with Query Rewriting, Multi-Query, and Reranking
(Groq LLM + FastEmbed backend)
"""

from typing import List, Dict

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from basic_rag import BasicRAG


class AdvancedRAG(BasicRAG):
    """
    Advanced RAG with query enhancement and reranking capabilities,
    using Groq LLM under the hood.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        query_limit: int = 100,
        max_tokens: int = 300,
        groq_model: str = "llama-3.1-8b-instant",
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            query_limit=query_limit,
            max_tokens=max_tokens,
            groq_model=groq_model,
        )
        # Groq LLM for advanced chains
        self.llm = ChatGroq(
            model=self.groq_model,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

    def query_rewriting(self, query: str) -> str:
        """
        Rewrite user query for better retrieval.
        """
        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that rewrites user queries to improve information retrieval.
            
Original query: {query}

Rewrite this query to be more specific and better suited for semantic search.
Focus on key concepts and remove ambiguity.

Rewritten query:"""
        )

        chain = LLMChain(llm=self.llm, prompt=rewrite_prompt)
        rewritten = chain.run(query=query)

        print(f"\n📝 Original Query: {query}")
        print(f"📝 Rewritten Query: {rewritten}")

        return rewritten.strip()

    def multi_query_retrieval(
        self, query: str, num_queries: int = 3, k: int = 3
    ) -> List:
        """
        Generate multiple query variations and retrieve documents.
        """
        multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant that generates multiple search queries.
            
Generate {num_queries} different versions of the following question to retrieve relevant documents:

Original question: {query}

Provide the queries as a numbered list:"""
        )

        chain = LLMChain(llm=self.llm, prompt=multi_query_prompt)
        result = chain.run(query=query, num_queries=num_queries)

        queries = [query]  # Include original
        for line in result.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() and (
                "." in stripped or ")" in stripped or ":" in stripped
            ):
                q = stripped.split(".", 1)[-1]
                q = q.split(")", 1)[-1]
                q = q.split(":", 1)[-1].strip()
                if q:
                    queries.append(q)

        print(f"\n🔍 Multi-Query Retrieval:")
        print(f"Generated {len(queries)} query variations:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")

        all_docs = []
        seen_contents = set()

        for q in queries:
            docs = self.vectorstore.similarity_search(q, k=k)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

        print(f"✓ Retrieved {len(all_docs)} unique documents")
        return all_docs

    def hyde_retrieval(self, query: str, k: int = 3) -> List:
        """
        HyDE (Hypothetical Document Embeddings) retrieval.
        Generate a hypothetical answer, then search for similar documents.
        """
        hyde_prompt = ChatPromptTemplate.from_template(
            """Generate a hypothetical detailed answer to the following question.
This answer will be used to find similar documents.

Question: {query}

Hypothetical Answer:"""
        )

        chain = LLMChain(llm=self.llm, prompt=hyde_prompt)
        hypothetical_answer = chain.run(query=query)

        print(f"\n🔮 HyDE Retrieval:")
        print(f"Generated hypothetical answer: {hypothetical_answer[:200]}...")

        docs = self.vectorstore.similarity_search(hypothetical_answer, k=k)
        print(f"✓ Retrieved {len(docs)} documents using HyDE")

        return docs

    def simple_rerank(self, query: str, documents: List, top_k: int = 3) -> List:
        """
        Simple reranking using LLM to score relevance.
        """
        print(f"\n🎯 Reranking {len(documents)} documents...")

        rerank_prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of the following document to the query on a scale of 0-10.
Only respond with a single number.

Query: {query}

Document: {document}

Relevance Score (0-10):"""
        )

        chain = LLMChain(llm=self.llm, prompt=rerank_prompt)

        scored_docs = []
        for doc in documents:
            try:
                score_str = chain.run(
                    query=query, document=doc.page_content[:500]
                )
                score = float(score_str.strip().split()[0])
                scored_docs.append((doc, score))
            except Exception:
                scored_docs.append((doc, 0.0))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        print("Reranking scores:")
        for i, (doc, score) in enumerate(scored_docs[:top_k], 1):
            print(f"  {i}. Score: {score}/10")

        return [doc for doc, score in scored_docs[:top_k]]

    def advanced_query(
        self,
        question: str,
        strategy: str = "multi_query",
        k: int = 3,
        rerank: bool = True,
    ) -> Dict:
        """
        Query with advanced retrieval strategies.
        """
        print(f"\n{'=' * 80}")
        print("ADVANCED RAG QUERY")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'=' * 80}")
        print(f"Question: {question}")

        if strategy == "multi_query":
            docs = self.multi_query_retrieval(question, num_queries=3, k=k)
        elif strategy == "hyde":
            docs = self.hyde_retrieval(question, k=k * 2)
        elif strategy == "query_rewrite":
            rewritten_q = self.query_rewriting(question)
            docs = self.vectorstore.similarity_search(rewritten_q, k=k * 2)
        else:
            docs = self.vectorstore.similarity_search(question, k=k)

        if rerank and len(docs) > k:
            docs = self.simple_rerank(question, docs, top_k=k)

        context = "\n\n".join([doc.page_content for doc in docs[:k]])

        answer_prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the question.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Detailed Answer:"""
        )

        chain = LLMChain(llm=self.llm, prompt=answer_prompt)
        answer = chain.run(context=context, question=question)

        print(f"\n{'=' * 80}")
        print("ANSWER:")
        print(f"{'=' * 80}")
        print(answer)

        print(f"\n{'=' * 80}")
        print(f"SOURCES ({len(docs[:k])}):")
        print(f"{'=' * 80}")
        for i, doc in enumerate(docs[:k], 1):
            print(f"\n[Source {i}]")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

        return {
            "result": answer,
            "source_documents": docs[:k],
            "strategy": strategy,
        }
