import sys
import json
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QListWidget,
    QListWidgetItem, QFileDialog, QComboBox, QMessageBox, QSplitter
)
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm import get_llm
from src.rag_pipeline import answer_question
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import build_vector_store

CHAT_FILE = Path.home() / "NexaAI" / "chats.json"


class AIWorker(QThread):
    answer_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, question, mode, history):
        super().__init__()
        self.question = question
        self.mode = mode
        self.history = history

    def run(self):
        try:
            if self.mode == "Documents":
                result = answer_question(self.question)
                answer = str(result.get("answer", "No answer returned."))
                sources = result.get("sources", [])
                if sources:
                    answer += "\n\nSources:\n" + "\n".join(
                        f"- {source}" for source in sources
                    )
                self.answer_ready.emit(answer)
                return

            messages = [
                SystemMessage(content=(
                    "You are Nexa AI, a helpful, accurate general-purpose "
                    "assistant. Answer in clear English. Be concise for "
                    "simple questions and structured for technical questions."
                ))
            ]
            messages.extend(self.history[-10:])
            messages.append(HumanMessage(content=self.question))
            response = get_llm().invoke(messages)
            self.answer_ready.emit(str(response.content))
        except Exception as exc:
            self.failed.emit(str(exc))


class NexaAIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nexa AI")
        self.resize(1200, 780)
        self.setMinimumSize(850, 600)

        self.chats = {}
        self.active_id = None
        self.worker = None
        self.selected_pdf = None
        self.load_chats()
        if not self.chats:
            self.new_chat()
        else:
            self.active_id = next(reversed(self.chats))

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 22, 16, 18)
        side.setSpacing(12)

        brand = QLabel("Nexa AI")
        brand.setObjectName("brand")
        side.addWidget(brand)

        new_button = QPushButton("+  New Chat")
        new_button.clicked.connect(self.new_chat)
        side.addWidget(new_button)

        side.addWidget(QLabel("RECENT CHATS"))
        self.chat_list = QListWidget()
        self.chat_list.currentItemChanged.connect(self.switch_chat)
        side.addWidget(self.chat_list, 1)

        self.upload_button = QPushButton("Upload PDF")
        self.upload_button.clicked.connect(self.choose_pdf)
        side.addWidget(self.upload_button)

        self.pdf_label = QLabel("No PDF selected")
        self.pdf_label.setWordWrap(True)
        side.addWidget(self.pdf_label)

        self.mode = QComboBox()
        self.mode.addItems(["General AI", "Documents"])
        side.addWidget(QLabel("Chat mode"))
        side.addWidget(self.mode)

        clear_button = QPushButton("Clear Current Chat")
        clear_button.clicked.connect(self.clear_chat)
        side.addWidget(clear_button)

        main = QWidget()
        content = QVBoxLayout(main)
        content.setContentsMargins(24, 22, 24, 20)
        content.setSpacing(12)

        heading = QLabel("Your Personal AI Assistant")
        heading.setObjectName("heading")
        content.addWidget(heading)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        content.addWidget(self.chat_view, 1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Message Nexa AI...")
        self.input.returnPressed.connect(self.send_message)
        row.addWidget(self.input, 1)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        row.addWidget(self.send_button)
        content.addLayout(row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(main)
        splitter.setSizes([260, 940])
        outer.addWidget(splitter)
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #17191f; color: #f1f2f5;
                font-size: 14px;
            }
            QWidget#sidebar { background: #20222a; }
            QLabel#brand { font-size: 27px; font-weight: 800; }
            QLabel#heading { font-size: 20px; font-weight: 650; }
            QTextEdit, QLineEdit, QListWidget, QComboBox {
                background: #252832; color: #f1f2f5;
                border: 1px solid #414552; border-radius: 8px;
                padding: 9px;
            }
            QPushButton {
                background: #6658e8; color: white; border: none;
                border-radius: 8px; padding: 11px;
                font-weight: 600;
            }
            QPushButton:hover { background: #796cf0; }
            QPushButton:disabled { background: #555966; }
            QListWidget::item { padding: 9px; }
            QListWidget::item:selected { background: #45405f; }
        """)

        self.refresh_chat_list()
        self.render_chat()

    def load_chats(self):
        try:
            if CHAT_FILE.exists():
                data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
                self.chats = data.get("chats", {})
                for chat in self.chats.values():
                    chat["messages"] = [
                        tuple(message) for message in chat.get("messages", [])
                    ]
        except Exception as exc:
            print(f"Chat history could not be loaded: {exc}")
            self.chats = {}

    def save_chats(self):
        try:
            CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {"chats": self.chats, "active_id": self.active_id}
            CHAT_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as exc:
            QMessageBox.warning(self, "Chat history", str(exc))

    def new_chat(self):
        cid = str(uuid.uuid4())
        self.chats[cid] = {"title": "New Chat", "messages": []}
        self.active_id = cid
        if hasattr(self, "chat_list"):
            self.save_chats()
        if hasattr(self, "chat_list"):
            self.refresh_chat_list()
            self.render_chat()

    def refresh_chat_list(self):
        self.chat_list.blockSignals(True)
        self.chat_list.clear()
        for cid, chat in reversed(list(self.chats.items())):
            item = QListWidgetItem(chat["title"][:32])
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.chat_list.addItem(item)
            if cid == self.active_id:
                self.chat_list.setCurrentItem(item)
        self.chat_list.blockSignals(False)

    def switch_chat(self, current, previous=None):
        if current is None:
            return
        cid = current.data(Qt.ItemDataRole.UserRole)
        if cid in self.chats:
            self.active_id = cid
            self.render_chat()

    def render_chat(self):
        if not hasattr(self, "chat_view") or self.active_id not in self.chats:
            return
        lines = []
        for role, text in self.chats[self.active_id]["messages"]:
            label = "You" if role == "user" else "Nexa AI"
            lines.append(f"<p><b>{label}:</b> {self.escape(text)}</p>")
        self.chat_view.setHtml("".join(lines) or
            "<h2>Welcome to Nexa AI</h2><p>Ask a question to get started.</p>")

    @staticmethod
    def escape(text):
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n", "<br>"))

    def choose_pdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a document",
            "",
            "Supported documents (*.pdf *.txt)"
        )
        if not path:
            return

        self.upload_button.setEnabled(False)
        self.pdf_label.setText("Reading and indexing document...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            documents = load_documents(path)
            if not documents:
                raise ValueError("No readable text was found in this document.")

            chunks = split_documents(documents)
            if not chunks:
                raise ValueError("No text chunks could be created.")

            build_vector_store(chunks)
            self.selected_pdf = path
            self.pdf_label.setText(Path(path).name)
            self.mode.setCurrentText("Documents")

            QMessageBox.information(
                self,
                "Indexing complete",
                f"Document indexed successfully.\n"
                f"Pages/sections loaded: {len(documents)}\n"
                f"Text chunks indexed: {len(chunks)}\n\n"
                "You can now ask questions in Documents mode."
            )
        except Exception as exc:
            self.pdf_label.setText("Document indexing failed")
            QMessageBox.critical(
                self, "Document indexing error", str(exc)
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.upload_button.setEnabled(True)

    def clear_chat(self):
        if self.active_id in self.chats:
            self.chats[self.active_id]["messages"] = []
            self.save_chats()
            self.chats[self.active_id]["title"] = "New Chat"
            self.refresh_chat_list()
            self.render_chat()

    def send_message(self):
        question = self.input.text().strip()
        if not question or (self.worker and self.worker.isRunning()):
            return

        chat = self.chats[self.active_id]
        chat["messages"].append(("user", question))
        self.save_chats()
        if chat["title"] == "New Chat":
            chat["title"] = question[:32]

        history = []
        for role, text in chat["messages"][:-1]:
            if role == "user":
                history.append(HumanMessage(content=text))
            else:
                from langchain_core.messages import AIMessage
                history.append(AIMessage(content=text))

        self.input.clear()
        self.refresh_chat_list()
        self.render_chat()
        self.send_button.setEnabled(False)
        self.input.setEnabled(False)

        self.worker = AIWorker(
            question, self.mode.currentText(), history
        )
        self.worker.answer_ready.connect(self.show_answer)
        self.worker.failed.connect(self.show_error)
        self.worker.start()

    def show_answer(self, answer):
        self.chats[self.active_id]["messages"].append(("assistant", answer))
        self.save_chats()
        self.render_chat()
        self.finish_request()

    def show_error(self, error):
        self.chats[self.active_id]["messages"].append(
            ("assistant", "Error: " + error)
        )
        self.save_chats()
        self.render_chat()
        self.finish_request()

    def finish_request(self):
        self.send_button.setEnabled(True)
        self.input.setEnabled(True)
        self.input.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NexaAIWindow()
    window.show()
    sys.exit(app.exec())
