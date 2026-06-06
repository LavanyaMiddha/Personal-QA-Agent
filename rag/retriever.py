from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from rag.embeddings import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    def __init__(self, search_type: str = "similarity", top_k: int = 5):
        self.search_type = search_type
        self.top_k = top_k

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

        self.vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index=self.index
        )

        self.retriever = self._create_retriever()

    def _create_retriever(self):
        supported_types = {"similarity", "mmr"}
        if self.search_type not in supported_types:
            raise ValueError(f"Unsupported search type: {self.search_type}. Choose from {supported_types}")

        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.top_k}
        )

    def retrieve(self, query: str):
        return self.retriever.invoke(query)


if __name__ == "__main__":
    print("Testing Retriever...")
    retriever = Retriever(search_type="similarity", top_k=5)
    print("Retriever created successfully. Retrieving documents...")
    results = retriever.retrieve("What is configuration management? what are the tools used for configuration management?")
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)
        print("--")