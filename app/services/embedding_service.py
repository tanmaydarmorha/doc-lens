from langchain_core.documents import Document
from utils import get_embedding_model
from langchain_community.vectorstores import FAISS
from typing import List


def create_embeddings_and_save_to_vector_store(documents: List[Document], tracking_uuid: str)-> None:
    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(documents, embedding_model)
    index_path = f"./faiss_index/{tracking_uuid}.index"
    vector_store.save_local(index_path)

def retrieve_from_vector_store(tracking_uuid: str)-> FAISS:
    embedding_model = get_embedding_model()
    index_path = f"./faiss_index/{tracking_uuid}.index"
    vector_store= FAISS.load_local(index_path, embedding_model)
    return vector_store