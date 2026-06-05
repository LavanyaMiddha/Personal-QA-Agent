from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Check current state
print("Before:", index.describe_index_stats())

# Upsert 5 dummy vectors directly
dummy_vectors = [
    {"id": f"test-{i}", "values": [0.1] * 768}
    for i in range(5)
]

result = index.upsert(vectors=dummy_vectors)
print("Upsert result:", result)

import time
time.sleep(5)
print("After:", index.describe_index_stats())