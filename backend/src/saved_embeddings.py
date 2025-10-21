import faiss
import numpy as np

index = faiss.IndexFlatL2(len(embedding))
index.add(np.array([embedding], dtype=np.float32))
