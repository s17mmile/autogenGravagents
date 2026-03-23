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
from PySide6.QtCore import QObject, Signal, Qt, Slot

class AgentChatGUI(QMainWindow):
    def __init__(self, agent_chat, parent = None):
        super().__init__(parent)
        self.agent_chat = agent_chat
        self.setWindowTitle("AgentChat Console")

        self._messages: List[Dict[str, Any]] = []

        # Main widget and layout
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

        # Connect button signals
        self.btn_load.clicked.connect(self.on_load_config_clicked)
        self.btn_save.clicked.connect(self.on_save_config_clicked)
        self.message_list.itemDoubleClicked.connect(self.on_message_double_clicked)

        # # Timer to poll for new messages (if you don't yet have callbacks)
        # self.poll_timer = QTimer(self)
        # self.poll_timer.timeout.connect(self.poll_agent_messages)
        # self.poll_timer.start(poll_interval_ms)

    # Load a config from a given file
    def on_load_config_clicked(self):
        dialog = QFileDialog(self, "Select agent config file")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Text files (*.txt);;All files (*)")
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                path = selected_files[0]
                try:
                    self.agent_chat.load_config(path)
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Error loading config",
                        f"Could not load config:\n{e}",
                    )

    # Placeholder for config mgmt
    def on_save_config_clicked(self):
        QMessageBox.information(
            self,
            "Save config",
            "Save config is not implemented yet.",
        )

    # Message Reception from AgentChat
    # Expected format: {"agent_name": "...", "message": "...", ...}
    # TODO update format to match agent output?
    def addMessage(self, msg: Dict[str, Any]):
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

    # Function to see details of message by just JSON dumping it into a pop-up
    def on_message_double_clicked(self, item: QListWidgetItem):
        msg = item.data(Qt.UserRole)
        if msg is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Message details")
        layout = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)

        # Pretty-print JSON for now
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


# Separate GUI worker class for cleaner multithreading
# TODO maybe this one can be the handler that manages the GUI on main thread and the AgentChat on a second thread?
# Message exposal would remain as a signal-slot mechanism
class GuiWorker(QObject):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.window = None

    @Slot()
    def buildGUI(self):
        self.window = AgentChatGUI(self.signals, parent = None)
        self.window.show()

    # Forwards message from AgentChat to the GUI thread as signal-slot (thread-safe)
    @Slot(dict)
    def addMessage(self, msg):
        if self.window:  # In case of early signal emission
            self.window.addMessage(msg)



# Gui Signal specification
class GuiSignals(QObject):
    new_message = Signal(dict)

'''
GUI code from AgentChat
        # Placeholder setters for GUI
        self.guiThread = None
        self.guiWorker = None
        self.guiSignals = None

        # TODO move this to a separate GUI handler class that this exposes messages to.
        # Build a GUI if needed
        # GUI runs on a separate thread and receives messages through a thread-safe queue, so it should not interrupt the main flow of the conversation at all.
        self.hasGUI = makeGUI
        if makeGUI:
            import threading
            from PySide6.QtCore import QObject, Signal, QThread, Slot
            from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
            from flexibleAgents.GUI.agentChatGUI import GuiWorker, GuiSignals

            # PySide6 App must be built in main thread
            self.app = QApplication(sys.argv)

            # Instantiate Signal Type to be used to send msg to GUI from main thread
            self.guiSignals = GuiSignals()

            # Build new thread and move the GUI handling class over there to avoid thread blocking
            self.guiThread = QThread()
            self.guiWorker = GuiWorker(self.guiSignals)
            self.guiWorker.moveToThread(self.guiThread)

            # Connect thread startup
            self.guiThread.started.connect(self.guiWorker.buildGUI)
            self.guiSignals.new_message.connect(self.guiWorker.addMessage)
            
            # Start thread (non-blocking)
            self.guiThread.start()
'''











def main():
    # Replace DummyAgentChat with your real agentChat instance
    agent_chat = DummyAgentChat()

    # Init PySide6 app
    app = QApplication(sys.argv)

    # For demo, inject some fake data periodically
    def demo_feed():
        # agent_chat.push_message("planner", "Planning next steps…", thought="…")
        # agent_chat.push_message("executor", "Code executed successfully.", status="ok")
        window.addMessage({"agent_name": "carl", "message": "hello", "comments": "nothing", "rounds": 1})
        window.addMessage({"agent_name": "jim", "message": "yo", "comments": "testing", "rounds": 2})

    demo_timer = QTimer()
    demo_timer.timeout.connect(demo_feed)
    demo_timer.start(3000)

    window = AgentChatGUI(agent_chat)
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

