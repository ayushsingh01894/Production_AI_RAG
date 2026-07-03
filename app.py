"""
Production AI RAG Assistant — CLI entry point.
Run: python app.py
"""

import os
import sys

from config import settings
from rag.pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger("cli")

MENU = """
==========================================
        Production AI RAG Assistant
==========================================

 1. Create Index
 2. Upload PDF
 3. Upload Folder
 4. List PDFs
 5. Semantic Search
 6. Ask AI
 7. Metadata Search
 8. Update Document
 9. Delete Document
10. Delete Namespace
11. Show Statistics
12. Clear Chat History
13. Change Namespace
14. Exit

==========================================
"""


def print_sources(chunks):
    if not chunks:
        print("\nSources: (none found)\n")
        return
    print("\nSources")
    print("-" * 40)
    for c in chunks:
        page_info = f"Page {c.page}" if c.page else ""
        print(f"{c.source}  {page_info}")
        print(f"Similarity: {c.score:.2f}")
        print("-" * 20)


def main():
    print("Initializing Production AI RAG Assistant...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        print("Check your .env configuration (PINECONE_API_KEY, LLM keys, etc).")
        sys.exit(1)

    current_namespace = settings.retrieval.default_namespace
    print(f"Ready. Active namespace: '{current_namespace}'")

    while True:
        print(MENU)
        choice = input("Select an option: ").strip()

        try:
            if choice == "1":
                pipeline.manager.create_index_if_not_exists(dimension=pipeline.embedder.dimension)
                print("Index ready.")

            elif choice == "2":
                path = input("Enter PDF file path: ").strip()
                if not os.path.isfile(path):
                    print("File not found.")
                    continue
                count = pipeline.upload_pdf(path, namespace=current_namespace)
                print(f"Uploaded {count} chunks to namespace '{current_namespace}'.")

            elif choice == "3":
                folder = input("Enter folder path: ").strip()
                if not os.path.isdir(folder):
                    print("Folder not found.")
                    continue
                results = pipeline.upload_folder(folder, namespace=current_namespace)
                for fname, res in results.items():
                    print(f"  {fname}: {res}")

            elif choice == "4":
                stats = pipeline.get_statistics()
                print("\nNamespaces and vector counts:")
                for ns, count in stats["namespaces"].items():
                    print(f"  {ns}: {count} vectors")

            elif choice == "5":
                query = input("Search query: ").strip()
                results = pipeline.semantic_search(query, namespace=current_namespace)
                print_sources(results)
                for r in results:
                    print(f"\n> {r.text[:300]}...\n")

            elif choice == "6":
                question = input("Ask a question: ").strip()
                mode = input(
                    "Mode [default/teacher/interviewer/technical_expert/beginner/summarizer] (Enter=default): "
                ).strip() or "default"
                result = pipeline.ask(question, namespace=current_namespace, mode=mode)
                print("\nAnswer:\n")
                print(result.answer)
                print_sources(result.sources)
                cache_note = " (from cache)" if result.from_cache else ""
                print(f"\n[latency: {result.latency_seconds:.2f}s{cache_note}]")

            elif choice == "7":
                query = input("Search query: ").strip()
                field = input("Metadata field to filter (e.g. source): ").strip()
                value = input(f"Value for '{field}': ").strip()
                results = pipeline.semantic_search(
                    query, namespace=current_namespace, metadata_filter={field: {"$eq": value}}
                )
                print_sources(results)

            elif choice == "8":
                vector_id = input("Vector ID to update: ").strip()
                new_text = input("New text: ").strip()
                pipeline.uploader.update_vector(vector_id, new_text, namespace=current_namespace)
                print("Vector updated.")

            elif choice == "9":
                source = input("Source filename to delete: ").strip()
                pipeline.delete_document(source, namespace=current_namespace)
                print(f"Deleted all chunks for '{source}'.")

            elif choice == "10":
                confirm = input(f"Type YES to delete namespace '{current_namespace}': ").strip()
                if confirm == "YES":
                    pipeline.delete_namespace(current_namespace)
                    print("Namespace deleted.")
                else:
                    print("Cancelled.")

            elif choice == "11":
                stats = pipeline.get_statistics()
                print("\nStatistics")
                print("-" * 40)
                print(f"Total Vectors     : {stats['total_vectors']}")
                print(f"Index Dimension   : {stats['dimension']}")
                print(f"Embedding Model   : {stats['embedding_model']}")
                print(f"LLM Provider      : {stats['llm_provider']}")
                print(f"Namespaces        : {stats['namespaces']}")

            elif choice == "12":
                pipeline.clear_chat_history()
                print("Chat history cleared.")

            elif choice == "13":
                new_ns = input("Enter new namespace name: ").strip()
                if new_ns:
                    current_namespace = new_ns
                    print(f"Active namespace changed to '{current_namespace}'.")

            elif choice == "14":
                print("Goodbye!")
                break

            else:
                print("Invalid option.")

        except Exception as e:
            logger.error(f"Error handling menu option {choice}: {e}")
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
