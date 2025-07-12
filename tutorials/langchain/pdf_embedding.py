from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths relative to the script directory
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
PDF_FILE = DATA_DIR / "pdf" / "mgn_paper.pdf"

# file_path = "./data/pdf/mgn_paper.pdf"
loader = PyPDFLoader(PDF_FILE)

docs = loader.load()

# print(len(docs))

# print(f"{docs[0].page_content}\n")
# print(docs[0].metadata)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="deepseek-r1:8b",
#     reasoning=False,
#     temperature=0.0
# )

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")

vectors = [embeddings.embed_query(split.page_content) for split in all_splits]
# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(vectors)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

indices = np.arange(len(embedding)) + 1  # Create indices (1-based)
unique_indices = np.unique(indices)
cmap = plt.get_cmap('viridis', len(unique_indices))
norm = BoundaryNorm(np.arange(0.5, len(unique_indices)+1.5, 1), cmap.N)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                      c=indices, cmap=cmap, norm=norm,
                      alpha=0.7, edgecolors='w')
cbar = plt.colorbar(scatter, ticks=unique_indices)
cbar.set_label('Index + 1')
for i, (x, y) in enumerate(embedding):
    plt.text(x, y, str(i+1), 
             fontsize=8, 
             ha='right', va='bottom')
plt.tight_layout()
plt.savefig("./data/pdf/embedding_discrete.png")
plt.close()