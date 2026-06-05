from pinecone import Pinecone
from dotenv import load_dotenv
import os
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

stats = index.describe_index_stats()

print("Total vectors:", stats["total_vector_count"])
print("Namespaces:", stats["namespaces"])