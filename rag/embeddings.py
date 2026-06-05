from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class Embeddings:
    def __init__(self, model_name: str = "gemini-embedding-2-preview", model_provider: str ="google-genai", **kwargs):
        self.kwargs = kwargs
        self.model_name = model_name
        self.model_provider = model_provider
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.model_name, model_provider=self.model_provider, output_dimensionality=768, task_type="RETRIEVAL_DOCUMENT", **self.kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(documents)


if __name__ == "__main__":
    embedding_model = Embeddings()
    query_embedding = embedding_model.embed_query("What is federated learning?")
    document_embeddings = embedding_model.embed_documents(["Federated learning is a machine learning technique that allows models to be trained across multiple decentralized devices or servers while keeping the data localized."])
    print("Query Embedding:", query_embedding)
    print("Document Embeddings:", document_embeddings)