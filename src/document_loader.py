from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


def load_documents(file_path):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError("Document file not found.")

    suffix = file_path.suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    size = file_path.stat().st_size
    if size == 0:
        raise ValueError("The uploaded document is empty.")
    if size > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 10 MB size limit.")

    if suffix == ".pdf":
        with file_path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError("Invalid PDF file signature.")
        return PyPDFLoader(str(file_path)).load()

    try:
        file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("TXT files must use UTF-8 encoding.") from exc

    return TextLoader(str(file_path), encoding="utf-8").load()
