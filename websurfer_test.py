from dotenv import load_dotenv
import os

print("importing playwright, llmconfig and websurfer")
from playwright.async_api import Download
from autogen import LLMConfig
from autogen.agents.experimental import WebSurferAgent
print("imported")

from llmconfig import local_llm_config, commercial_llm_config

def download_handler(download: Download, context: str = None) -> None:
    if "corpus" in context:
        target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "downloadTest1"))
    elif "data" in context:
        target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "downloadTest1"))
    else:
        download.cancel()  # Block other downloads
        return
    
    os.makedirs(target_folder, exist_ok=True)
    download_path = os.path.join(target_folder, download.suggested_filename)
    download.save_as(download_path)


print("agent building")

agent = WebSurferAgent(
    name="WebSurfer",
    llm_config=local_llm_config,
    system_message="""
        You are a web surfing agent that can download files to specific folders based on the context of the conversation.
        You can use context "corpus" or "data" to determine where to save the downloaded files.
        "Corpus" is to be used for documents such as web pages or publications, including information to be used for RAG.
        "Data" is to be used for raw data files, such as CSVs, JSONs, or other structured data that can be used for analysis within the context of a single task.
        If the context is not clear, you should not download the file.
    """,
    web_tool="browser_use",
    web_tool_kwargs={
        "_download_handler": download_handler
    }
)

print("agent built")

agent.run(
    "Please download the english wikipedia page for 'Python' to the corpus folder and the raw data of event the GW150914 Gravitational Wave detection to the data folder."
)