import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# Knowledge Base
# ============================================================

knowledge_base = [
    "Forgotten passwords can be recovered through the password reset portal.",
    "Customers can raise billing disputes by contacting the finance support team.",
    "Personal information can be updated from the profile management section.",
    "Sign-in issues may result from invalid login credentials.",
    "Two-step verification helps secure user accounts from unauthorized access.",
    "Users can switch between subscription tiers whenever needed.",
    "Refund processing typically takes five to seven business days.",
    "A verified email address is required for account activation.",
    "Alert and notification settings can be managed in account preferences.",
    "Repeated violations of platform policies can lead to account suspension."
]

# ============================================================
# Load Embedding Model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# Generate Embeddings
# ============================================================

embeddings = model.encode(knowledge_base)

print("\nEmbedding Matrix Shape:")
print(embeddings.shape)

# Convert embeddings to float32 (required by FAISS)
embeddings = np.array(embeddings).astype("float32")

# ============================================================
# Normalize Embeddings
# ============================================================

faiss.normalize_L2(embeddings)

# ============================================================
# Build FAISS Index
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nTotal Vectors Stored:")
print(index.ntotal)

# ============================================================
# Semantic Search Function
# ============================================================

def semantic_search(query, top_k=3):
    """
    Performs semantic search using FAISS.
    """

    # Generate query embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    # Retrieve Top-K results
    distances, indices = index.search(query_embedding, top_k)

    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    print(f"{'Rank':<10}{'Score':<15}{'Matched Sentence'}")
    print("-" * 80)

    for rank, idx in enumerate(indices[0], start=1):

        score = distances[0][rank - 1]

        print(
            f"{rank:<10}{score:<15.4f}{knowledge_base[idx]}"
        )

# ============================================================
# Sample Queries
# ============================================================

print("\nRunning Sample Queries...\n")

semantic_search("How do I recover my password?")

semantic_search("When will my refund be processed?")

semantic_search("Why am I unable to sign in?")

# ============================================================
# Interactive CLI
# ============================================================

print("\n")
print("=" * 60)
print("FAISS Semantic Search Engine Ready")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    user_query = input("\nEnter your query: ")

    if user_query.lower() == "exit":
        print("\nExiting Semantic Search Engine...")
        break

    semantic_search(user_query)