from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from embeddings import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    def __init__(self, embeddings: Embeddings, search_type: str = "similarity", top_k: int = 5):
        self.search_type = search_type
        self.top_k = top_k

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embeddings.model_name,
            output_dimensionality=768,
            task_type="RETRIEVAL_QUERY"   
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
    embeddings = Embeddings()
    retriever = Retriever(embeddings=embeddings, search_type="mmr", top_k=5)
    results = retriever.retrieve("What is configuration management? what are the tools used for configuration management?")
    for doc in results:
        print(doc.page_content)
        print(doc.metadata)
        print("--")