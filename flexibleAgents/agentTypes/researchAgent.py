import os
from pydantic import BaseModel
from typing import Dict, List

print("importing playwright and websurfer")
from playwright.async_api import Download
from autogen.agents.experimental import WebSurferAgent
print("imported")


# Define research agent response format
class researchAgentResponse(BaseModel):
	message: str								# Overview of found documents/visited web pages
	foundDocumentNames: List[str]			    # List of names of documents (so the human can cross-check sources)

# Custom download management to enable precise downloading of data and corpus documents
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



# The Research Agent is responsible for conducting research on the web to find relevant information and sources to answer queries posed by other agents.
# It is responsible for finding and downloading relevant documents from the web, which can then be ingested into the system and used by the Document Retrieval Agent to answer questions with verifiable sources.
def researchAgent(llm_config, name = "ResearchAgent") -> WebSurferAgent:
	# Define path for document corpus
	documentCorpusPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentCorpus"))

	systemMessage = f"""
		You are a RESEARCH AGENT specializing in finding and downloading relevant documents or data from the web.
          
        You can use context "corpus" or "data" to determine where to save the downloaded files.
        "Corpus" is to be used for documents such as web pages or publications (HTML, Text Files, PDFs, etc.). These will be ingested into the Document Corpus used for Retrieval Augmented Generation.
        "Data" is to be used for raw data files, such as CSVs, JSONs, or other structured data that can be used for analysis within the context of a single task.
        If the context is not clear, you should not download the file.
		
		Your responsibilities:
		1. Search the web for relevant information based on the queries you receive.
        2. Download any informative documents and store them in the local document corpus for use by the Document Retrieval Agent.
		3. Do not perform any reasoning or answering of questions yourself - your sole responsibility is to find and download relevant documents from the web to be ingested into the system.

		Your output includes a message field and a foundDocumentNames field:
		- The message field should contain an overview of the pages visited and files downloaded to answer the query.
		- The foundDocumentNames field should list the names of the documents you found and downloaded to support your answer.
	"""

	description = """
		The RESEARCH AGENT is responsible for finding and downloading relevant documents from the web to be ingested into the system.
		It should find and download documents that can be used by the Document Retrieval Agent to answer queries with verifiable sources.
	"""

	research_llm_config = llm_config.copy()
	research_llm_config["response_format"] = researchAgentResponse
	research_llm_config["temperature"] = 0.01

	# Using a given collection name is needed to retain knowledge across runs
	return WebSurferAgent(
		name = name,
		system_message = systemMessage,
		llm_config = research_llm_config,
		web_tool="browser_use",
		web_tool_kwargs={
        	"_download_handler": download_handler
    	}
	)
