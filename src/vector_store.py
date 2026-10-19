from langchain_community.vectorstores import FAISS
from src.config import VECTOR_STORE_DIR
from src.embeddings import get_embeddings


def build_vector_store(documents):
    if not documents:
        raise ValueError("No document chunks were provided for indexing.")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    return vector_store


def get_vector_store():
    index_path = VECTOR_STORE_DIR / "index.faiss"
    metadata_path = VECTOR_STORE_DIR / "index.pkl"

    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "No complete local index found. Upload and index a PDF or TXT file first."
        )

    # Only load indexes created and controlled by this application.
    # Never load FAISS indexes or pickle files from untrusted sources.
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
