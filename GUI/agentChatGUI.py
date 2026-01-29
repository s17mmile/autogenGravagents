import sys
import json
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QDialog,
    QTextEdit,
)
from PySide6.QtCore import QTimer, Qt


class AgentChatGUI(QMainWindow):
    def __init__(self, agent_chat, poll_interval_ms: int = 500, parent=None):
        """
        agent_chat: an instance of your agentChat class
        poll_interval_ms: how often to poll for new messages (if using polling)
        """
        super().__init__(parent)
        self.agent_chat = agent_chat
        self.setWindowTitle("AgentChat Console")

        self._messages: List[Dict[str, Any]] = []

        # ---- Main widget and layout ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top button row
        button_row = QHBoxLayout()
        self.btn_load = QPushButton("Load config…")
        self.btn_save = QPushButton("Save config… (placeholder)")
        button_row.addWidget(self.btn_load)
        button_row.addWidget(self.btn_save)
        button_row.addStretch()

        main_layout.addLayout(button_row)

        # Chat message list
        self.message_list = QListWidget()
        self.message_list.setSelectionMode(QListWidget.SingleSelection)
        main_layout.addWidget(self.message_list)

        # Connect signals
        self.btn_load.clicked.connect(self.on_load_config_clicked)
        self.btn_save.clicked.connect(self.on_save_config_clicked)
        self.message_list.itemDoubleClicked.connect(self.on_message_double_clicked)

        # Timer to poll for new messages (if you don't yet have callbacks)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_agent_messages)
        self.poll_timer.start(poll_interval_ms)

    # ---------- Button handlers ----------

    def on_load_config_clicked(self):
        dialog = QFileDialog(self, "Select agent config file")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Text files (*.txt);;All files (*)")
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                path = selected_files[0]
                try:
                    # Assumes your class has this method:
                    # e.g., self.agent_chat.load_config(path)
                    self.agent_chat.load_config(path)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error loading config",
                        f"Could not load config:\n{e}",
                    )

    def on_save_config_clicked(self):
        # Placeholder; you can later wire this to e.g. self.agent_chat.save_config(path)
        QMessageBox.information(
            self,
            "Save config",
            "Save config is not implemented yet.",
        )

    # ---------- Message handling ----------

    def poll_agent_messages(self):
        """
        Example polling method: adapt to how your agentChat exposes new messages.
        For example, you might have agent_chat.get_new_messages() that returns a list.
        """
        try:
            if hasattr(self.agent_chat, "get_new_messages"):
                # You define this API; here we assume it returns a list of dicts
                new_msgs = self.agent_chat.get_new_messages()
            else:
                new_msgs = []
        except Exception as e:
            # Optional: log or show transient error message
            print(f"Error polling messages: {e}")
            return

        if not new_msgs:
            return

        for msg in new_msgs:
            self.add_message(msg)

    def add_message(self, msg: Dict[str, Any]):
        """
        Add a single message dict to the list widget and store its full data.
        Expected format: {"agent_name": "...", "message": "...", ...}
        """
        agent_name = msg.get("agent_name", "agent")
        message_text = msg.get("message", "")

        list_text = f"[{agent_name}] {message_text}"
        item = QListWidgetItem(list_text)

        # Store full message dict on the item for later retrieval
        item.setData(Qt.UserRole, msg)

        self.message_list.addItem(item)
        self._messages.append(msg)

        # Scroll to bottom
        self.message_list.scrollToBottom()

    # ---------- Detail dialog ----------

    def on_message_double_clicked(self, item: QListWidgetItem):
        msg = item.data(Qt.UserRole)
        if msg is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Message details")
        layout = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)

        # Pretty-print JSON for now; later you can build custom field views
        try:
            pretty = json.dumps(msg, indent=2, ensure_ascii=False)
        except TypeError:
            pretty = str(msg)

        text.setPlainText(pretty)
        layout.addWidget(text)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)

        dlg.resize(500, 400)
        dlg.exec()


# ---------- Example integration ----------

# This is just a stub to demonstrate expected methods; replace with your real class.
class DummyAgentChat:
    def __init__(self):
        self._buffer: List[Dict[str, Any]] = []

    def load_config(self, path: str):
        print(f"Loading config from {path!r}")
        # Implement your actual config loading here

    def get_new_messages(self) -> List[Dict[str, Any]]:
        """
        In your real implementation, you might:
        - pull from an internal queue,
        - read from AG2 conversation logs,
        - or receive callbacks from agents and buffer them.
        """
        msgs = self._buffer
        self._buffer = []
        return msgs

    # For demo: push a message into the buffer from outside
    def push_message(self, agent_name: str, message: str, **kwargs):
        data = {"agent_name": agent_name, "message": message}
        data.update(kwargs)
        self._buffer.append(data)


def main():
    app = QApplication(sys.argv)

    # Replace DummyAgentChat with your real agentChat instance
    agent_chat = DummyAgentChat()

    # For demo, inject some fake data periodically
    def demo_feed():
        agent_chat.push_message("planner", "Planning next steps…", thought="…")
        agent_chat.push_message("executor", "Code executed successfully.", status="ok")

    demo_timer = QTimer()
    demo_timer.timeout.connect(demo_feed)
    demo_timer.start(3000)

    window = AgentChatGUI(agent_chat)
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

