"""
Testing and Evaluation Module for RAG Pipeline (Groq LLM + FastEmbed)
"""

import time
import json
from typing import List, Dict

from basic_rag import BasicRAG
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


class RAGEvaluator:
    """
    Evaluate RAG pipeline performance.
    Uses Groq LLM both for baseline and evaluation prompts.
    """

    def __init__(self, rag_system: BasicRAG, groq_model: str = "llama-3.1-8b-instant"):
        self.rag = rag_system
        self.llm = ChatGroq(model=groq_model, temperature=0.0, max_tokens=300)
        self.results = []

    def compare_with_baseline(self, question: str) -> Dict:
        """
        Compare RAG answer with baseline LLM (no retrieval).
        """
        print(f"\n{'=' * 80}")
        print("COMPARISON TEST")
        print(f"{'=' * 80}")
        print(f"Question: {question}")

        print("\n[1] RAG System Answer:")
        print("-" * 80)
        start = time.time()
        rag_response = self.rag.query(question, verbose=False)
        rag_time = time.time() - start
        print(rag_response["result"])

        print("\n[2] Baseline LLM Answer (No RAG):")
        print("-" * 80)
        start = time.time()
        baseline_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_template("{question}"),
        )
        baseline_response = baseline_chain.run(question=question)
        baseline_time = time.time() - start
        print(baseline_response)

        print(f"\n⏱️  RAG Time: {rag_time:.2f}s | Baseline Time: {baseline_time:.2f}s")

        return {
            "question": question,
            "rag_answer": rag_response["result"],
            "baseline_answer": baseline_response,
            "rag_sources": len(rag_response["source_documents"]),
            "rag_time": rag_time,
            "baseline_time": baseline_time,
        }

    def evaluate_faithfulness(
        self, question: str, answer: str, sources: List
    ) -> Dict:
        """
        Evaluate if answer is faithful to source documents.
        """
        source_context = "\n\n".join([doc.page_content for doc in sources])

        faithfulness_prompt = ChatPromptTemplate.from_template(
            """Evaluate if the answer is faithful to the provided context.
            
Context:
{context}

Question: {question}

Answer: {answer}

Is the answer supported by the context? Rate 1-5:
1 = Not supported at all (hallucination)
2 = Partially supported
3 = Mostly supported
4 = Well supported
5 = Fully supported with evidence

Respond with just the number and a brief explanation."""
        )

        chain = LLMChain(llm=self.llm, prompt=faithfulness_prompt)
        evaluation = chain.run(
            context=source_context[:2000], question=question, answer=answer
        )

        return {
            "faithfulness_score": evaluation,
            "sources_used": len(sources),
        }

    def evaluate_relevance(self, question: str, answer: str) -> str:
        """
        Evaluate answer relevance to question.
        """
        relevance_prompt = ChatPromptTemplate.from_template(
            """Rate how relevant and helpful this answer is to the question.
            
Question: {question}

Answer: {answer}

Rate 1-5:
1 = Not relevant
2 = Slightly relevant
3 = Moderately relevant
4 = Very relevant
5 = Perfectly relevant and complete

Respond with just the number and brief explanation."""
        )

        chain = LLMChain(llm=self.llm, prompt=relevance_prompt)
        return chain.run(question=question, answer=answer)

    def run_test_suite(self, test_cases: List[Dict]) -> None:
        """
        Run comprehensive test suite.
        """
        print("\n" + "=" * 80)
        print("RUNNING TEST SUITE")
        print("=" * 80)

        results = []

        for i, test in enumerate(test_cases, 1):
            print(f"\n{'#' * 80}")
            print(f"Test Case {i}/{len(test_cases)}")
            print(f"{'#' * 80}")

            question = test["question"]
            expected_type = test.get("type", "factual")

            start = time.time()
            response = self.rag.query(question, verbose=False)
            elapsed = time.time() - start

            print(f"\nQuestion: {question}")
            print(f"Type: {expected_type}")
            print(f"\nAnswer:\n{response['result']}")

            faithfulness = self.evaluate_faithfulness(
                question,
                response["result"],
                response["source_documents"],
            )

            relevance = self.evaluate_relevance(
                question,
                response["result"],
            )

            result = {
                "test_case": i,
                "question": question,
                "type": expected_type,
                "answer": response["result"],
                "response_time": elapsed,
                "num_sources": len(response["source_documents"]),
                "faithfulness": faithfulness,
                "relevance": relevance,
            }

            results.append(result)

            print(f"\n📊 Faithfulness: {faithfulness['faithfulness_score']}")
            print(f"📊 Relevance: {relevance}")
            print(f"⏱️  Response Time: {elapsed:.2f}s")

            if i < len(test_cases):
                input("\nPress Enter for next test...")

        self.save_results(results)
        self.print_summary(results)

    def save_results(self, results: List[Dict], filename: str = "test_results.json"):
        """Save test results to file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {filename}")

    def print_summary(self, results: List[Dict]):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        avg_time = sum(r["response_time"] for r in results) / len(results)
        avg_sources = sum(r["num_sources"] for r in results) / len(results)

        print(f"\nTotal Tests: {len(results)}")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Average Sources Used: {avg_sources:.1f}")


def create_test_cases() -> List[Dict]:
    """Create test cases for evaluation."""
    return [
        {
            "question": "What is the main topic discussed in the documents?",
            "type": "factual",
            "category": "in_domain",
        },
        {
            "question": "Can you summarize the key findings?",
            "type": "summarization",
            "category": "in_domain",
        },
        {
            "question": "What methodology was used?",
            "type": "factual",
            "category": "in_domain",
        },
        {
            "question": "What are the limitations mentioned?",
            "type": "analytical",
            "category": "in_domain",
        },
        {
            "question": "Who is the president of Mars?",
            "type": "out_of_scope",
            "category": "out_of_domain",
        },
        {
            "question": "Compare the results across different sections",
            "type": "comparative",
            "category": "in_domain",
        },
    ]


def demo_evaluation():
    """Demo evaluation functionality."""
    print("=" * 80)
    print("RAG EVALUATION DEMO (Groq)")
    print("=" * 80)

    rag = BasicRAG()
    rag.load_vectorstore("./chroma_db")
    rag.setup_chain(k=3, temperature=0.0)

    evaluator = RAGEvaluator(rag)

    test_question = "What is the main topic of the document?"
    evaluator.compare_with_baseline(test_question)

    print("\n\n")

    test_cases = create_test_cases()
    evaluator.run_test_suite(test_cases)


if __name__ == "__main__":
    demo_evaluation()
