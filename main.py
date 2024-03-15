
# pip install llama-index
# pip install llama-index-embeddings-huggingface

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)


retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
)
# retriever = index.as_retriever(similarity_top_k=2)


embeddings = Settings.embed_model.get_text_embedding(
    ["It is raining cats and dogs here!", "another"]
)
