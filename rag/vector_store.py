from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import uuid
import time
import os

from data_loader import DataLoader

load_dotenv()


class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self._create_index()
        self.index = self.pc.Index(self.index_name)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index=self.index
        )

    def _create_index(self):
        existing_indexes = [
            index["name"]
            for index in self.pc.list_indexes()
        ]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,          
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created index '{self.index_name}'")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def clear_index(self):
        self.index.delete(delete_all=True)
        print("Index cleared.")
        time.sleep(5)

    def add_documents(self, documents: list, batch_size: int = 50):
        print(f"Attempting to upsert {len(documents)} splits")
        all_results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [str(uuid.uuid4()) for _ in batch]

            try:
                results = self.vector_store.add_documents(batch, ids=ids)
                all_results.extend(results)
                print(f"Batch {i//batch_size + 1} OK — running total: {len(all_results)}")
            except Exception as e:
                print(f"Batch {i//batch_size + 1} FAILED: {e}")
                raise e

        print("Waiting for Pinecone to finish indexing...")
        prev_count = 0
        for attempt in range(20):
            time.sleep(5)
            stats = self.index.describe_index_stats()
            current_count = stats.total_vector_count
            print(f"  Attempt {attempt + 1}: {current_count} records indexed")
            if current_count == prev_count and current_count > 0:
                print("Count stabilized.")
                break
            prev_count = current_count

        print(f"Final record count in Pinecone: {stats.total_vector_count}")
        return all_results


if __name__ == "__main__":
    data_loader = DataLoader()
    documents = data_loader.load_pdfs("../data/pdf_files")
    splits = data_loader.return_splits(documents)

    vector_store = VectorStore()
    #vector_store.clear_index()
    vector_store.add_documents(splits)

    print("Documents added successfully!")