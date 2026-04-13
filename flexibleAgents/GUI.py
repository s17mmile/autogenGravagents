import sys
import json
from typing import Optional, List, Dict, Any
import threading

from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
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
from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal, Qt, Slot, QThread

from flexibleAgents.agentChat import flexibleAgentChat


# Define signal types for new messages and chat starts
class GUISignals(QObject):
    sendQuery = Signal(str)
    loadConfigRequest = Signal(str)
    buildAgentsRequest = Signal()
    interruptRequest = Signal()


class AgentChatGUI(QMainWindow):

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # Definition of GUI itself (closed off from agent chat logic and message handling for cleaner code organization)

    def __init__(self, agentChat: flexibleAgentChat, parent = None):
        super().__init__(parent)
        self.setWindowTitle("AgentChat Console")

        # GUI knowledge about the agent chat and communication signals
        self.agentChat = agentChat
        self.messages: List[Dict[str, Any]] = []
        self.signals = GUISignals()

        # The GUI shutdown process needs to be able to call a handler function in the main thread, so the handler is registered.
        self.handler = None

        # Main widget and layout
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.main_layout = QVBoxLayout(self.central)

        # Top button row
        self.button_row = QHBoxLayout()
        self.btn_load = QPushButton("Load config…")
        self.btn_build = QPushButton("Build agents")
        self.btn_save = QPushButton("Save config… (placeholder)")
        self.button_row.addWidget(self.btn_load)
        self.button_row.addWidget(self.btn_build)
        self.button_row.addWidget(self.btn_save)
        self.button_row.addStretch()

        # Chat message list
        self.message_list = QListWidget()
        self.message_list.setSelectionMode(QListWidget.SingleSelection)

        # Input row at bottom
        self.input_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query here...")
        self.btn_send = QPushButton("Send query")
        self.btn_interrupt = QPushButton("Interrupt chat")
        self.input_row.addWidget(self.query_input)
        self.input_row.addWidget(self.btn_send)
        self.input_row.addWidget(self.btn_interrupt)
        self.input_row.addStretch()

        # Add everything to the main layout
        self.main_layout.addLayout(self.button_row)
        self.main_layout.addWidget(self.message_list)
        self.main_layout.addLayout(self.input_row)

        # Button and Signal/slot connections
        self.connectChatSignals()
        self.connectButtons()

    def registerHandler(self, handler):
        self.handler = handler

    # Connect button signals
    def connectButtons(self):
        self.btn_load.clicked.connect(self.on_load_config_clicked)
        self.btn_save.clicked.connect(self.on_save_config_clicked)
        self.btn_build.clicked.connect(self.on_build_agents_clicked)
        self.btn_send.clicked.connect(self.on_send_query_clicked)
        self.btn_interrupt.clicked.connect(self.on_interrupt_chat_clicked)
        self.message_list.itemDoubleClicked.connect(self.on_message_double_clicked)

        # Hitting enter in the query input also sends the query
        self.query_input.returnPressed.connect(self.on_send_query_returnPressed)

    # Connect signals for agent chat control through GUI
    def connectChatSignals(self):
        self.signals.sendQuery.connect(self.agentChat.startConversation)
        self.signals.loadConfigRequest.connect(self.agentChat.parseAgentConfig)
        self.signals.buildAgentsRequest.connect(self.agentChat.buildAgents)
        self.signals.interruptRequest.connect(self.agentChat.interruptChat)

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
                    print("Emitting load config signal with path:", path)
                    self.signals.loadConfigRequest.emit(path)
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

    def on_build_agents_clicked(self):
        try:
            self.signals.buildAgentsRequest.emit()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error sending building agents signal",
                f"Could not build agents:\n{e}",
            )

    def on_send_query_clicked(self):
        # Don't send empty queries
        if self.query_input.text().strip() == "":
            print("Empty query, not sending.")
            return
        else:
            try:
                self.signals.sendQuery.emit(self.query_input.text())
                self.query_input.clear()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error sending query signal",
                    f"Could not send query:\n{e}",
                )

    def on_send_query_returnPressed(self):
        self.on_send_query_clicked()

    def on_interrupt_chat_clicked(self):
        try:
            print("Interrupt button clicked, emitting interrupt signal. Will be processed once thread is free.")
            self.signals.interruptRequest.emit()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error sending interrupt signal",
                f"Could not send interrupt signal:\n{e}",
            )

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

    # --------------------------------------------------------------------------------------------------------------------------------------------------
    # GUI Interaction with handler and chat

    # Message Reception from AgentChat
    # Expected format: {"name": "...", "content": {JSON_STRING}, ...}
    @Slot(dict)
    def addMessage(self, msg: Dict[str, Any]):
        print("receiving message in GUI:", msg)

        agent_name = msg["name"] if "name" in msg else "Unknown"
        message_text = msg["content"]["message"] if "content" in msg and "message" in msg["content"] else "ERROR: No message content"

        list_text = f"[{agent_name}] {message_text}"
        item = QListWidgetItem(list_text)

        # Store full message dict on the item for later retrieval
        item.setData(Qt.UserRole, msg)

        self.message_list.addItem(item)
        self.messages.append(msg)

        # Scroll to bottom
        self.message_list.scrollToBottom()

        print("Message added to GUI.")

    @Slot(str)
    def showPopup(self, text: str):
        QMessageBox.information(
            self,
            "Agent Chat",
            text,
        )

    # Shutdown handling overrides regular closing functionality with handler thread cleanup before close
    def closeEvent(self, event):
        self.handler.threadCleanup()
        event.accept()

# Separate GUI worker class for cleaner multithreading
# Message exposal would remain as a signal-slot mechanism
class AgentChatGuiHandler(QObject):
    def __init__(self, configPath: str = None, llm_config = None, maxRounds: int = 10):
        super().__init__()
        self.app = QApplication(sys.argv)

        # Build the agentic chat. Will be moved to a separate thread but needs to be built before the GUI so that it can be registered to the GUI signals.
        # ConfigPath is not yet passed sa GUI registration should happen before - didn't quite build this as cleanly as I may have liked.
        # Could have made a GUI parameter passable to flexibleAgentChat but preferred separate GUI registration to keep backend clean.
        self.agentChatThread = QThread()
        self.agentChat = flexibleAgentChat(None, llm_config=llm_config, maxRounds=maxRounds)

        # Guild agents if config is provided
        if configPath:
            self.agentChat.parseAgentConfig(configPath)
            self.agentChat.buildAgents()

        # Build Agent Chat GUI
        self.window = AgentChatGUI(self.agentChat, parent = None)
        self.window.registerHandler(self)
        self.window.show()
        self.window.resize(800, 600)

        # Register GUI to agent chat for bidirectional signal-slot communication
        self.agentChat.registerGUI(self.window)
        
        # Run PySide6 App on main thread while the agentChat lives on a second thread to avoid blocking. Communication happens through signals/slots.
        self.agentChat.moveToThread(self.agentChatThread)
        self.agentChatThread.start()

        sys.exit(self.app.exec())



    def threadCleanup(self):
        print("Handler: Window closing, stopping agentChat thread...")
        
        # Emit chat interrupt request
        self.window.signals.interruptRequest.emit()
        
        # Queue quit signal
        self.agentChatThread.quit()
        
        # Pop-up window to inform about force-shutdown timer (memory corruption is possible if chat is currently executing).
        msgBox = QMessageBox(self.window)
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle("Shutdown")
        msgBox.setText("Shutting down the agent chat thread, 3s until forced shutdown.")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.show()

        # Wait max 3s for clean exit (processing of the previous quit signal, will only happen if agent message finishes in this time).
        # If quit signal is not processed in this time, it means the agent is busy (most likely just waiting for a chat completion response) and a stop is forced. Extra wait time for cleanup.
        if not self.agentChatThread.wait(3000):
            print("Force terminating agentChatThread...")
            self.agentChatThread.terminate()
            self.agentChatThread.wait(500)

        msgBox.close()

        print("Thread shutdown complete.")