import os
import shutil
from autogen import ConversableAgent
from browser_use import BrowserContextConfig
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Union

print("importing websurfer")
from autogen.agents.experimental import WebSurferAgent
print("imported")

# Note: the documentCollectionAgent is based on websurferagent, which is experimental and causes a few warnings internally:
# - a pydantic warning about an improper field
# - a future warning about google genai. 
# - a warning about the "stream" parameter
# - a warning about pkg_resources deprecation
# This is out of my control and does not affect execution.

# Download specification for research agent output
class downloadSpecification(BaseModel):
	filename: str
	destination: Literal["corpus", "data"]

# Define document collection agent response format
class documentCollectionAgentResponse(BaseModel):
	message: str									# Overview of found documents/visited web pages, important takeaways
	filesDownloaded: List[downloadSpecification]	# List of names and destinations of downloaded documents

# This function moves the downloaded documents to their respective folders based on the document collection Agent's "destination" specification.
def sortDocuments(sender: ConversableAgent, message: Union[dict[str, Any], str], recipient: ConversableAgent, silent: bool):
	try:
		content = message.get("content", {})
		filesDownloaded = content.get("filesDownloaded", [])
		for file in filesDownloaded:
			filename = file["filename"]
			destination = file["destination"]

			sourcePath = os.path.join(os.path.dirname(__file__), "../tempConversation/tempDownloads", filename)

			if destination == "corpus":
				destDir = os.path.join(os.path.dirname(__file__), "../documentCorpus", filename)
			elif destination == "data":
				destDir = os.path.join(os.path.dirname(__file__), "../data", filename)
			else:
				continue

			shutil.move(sourcePath, destDir)

		return message
	
	except Exception as e:
		print(f"Error sorting documents: {e}")
		return message



# The Document Collection Agent is responsible for collecting and organizing relevant documents and data from the web.
# It is responsible for finding and downloading relevant documents from the web, which can then be ingested into the system and used by the Document Retrieval Agent to answer questions with verifiable sources.
def documentCollectionAgent(llm_config, name = "DocumentCollectionAgent") -> WebSurferAgent:
	systemMessage = f"""
		You are a DOCUMENT COLLECTION AGENT specializing in finding and downloading relevant documents or data from the web.
          
		Your responsibilities:
		1. Search the web for relevant information based on the queries you receive.
        2. Download any informative documents and store them in the local document corpus for use by the Document Retrieval Agent.
		3. Download any relevant data files (CSVs, JSONs, etc.) that can be used for analysis within the context of a single task.
		4. Do not perform any reasoning or answering of questions yourself - your sole responsibility is to find and download relevant documents from the web to be ingested into the system.

		Your downloads will go to a temporary download folder. With each download, specify the name of the file and whether it should be moved to the "corpus" (for documents) or "data" (for raw data files) folder. This file movement is handled automatically.
		Avoid Captchas or similar roadblocks - if you encounter a Captcha, go back try to find another source for the same information.
		
		Your output includes a message field and a filesDownloaded field:
		- The message field should contain an overview of the pages visited and important takeaways to answer the query.
		- The filesDownloaded field should list the names (including file extensions) and destinations ("corpus" or "data") of the documents you found and downloaded.
        	- "Corpus" is to be used for documents such as web pages or publications (HTML, Text Files, PDFs, etc.). These will be ingested into the Document Corpus used for Retrieval Augmented Generation.
        	- "Data" is to be used for raw data files, such as CSVs, JSONs, or other structured data that can be used for analysis within the context of a single task.
        	- If the purpose of a file is not clear, you should not download it.
	"""

	description = """
		The DOCUMENT COLLECTION AGENT is responsible for finding and downloading relevant documents from the web to be ingested into the system.
		It should find and download documents that can be used by the Document Retrieval Agent to answer queries with verifiable sources.
	"""

	documentCollection_llm_config = llm_config.copy()
	documentCollection_llm_config["response_format"] = documentCollectionAgentResponse
	documentCollection_llm_config["temperature"] = 0.01

	# Using a given collection name is needed to retain knowledge across runs
	agent = WebSurferAgent(
		name = name,
		system_message = systemMessage,
		llm_config = documentCollection_llm_config,
		web_tool="browser_use",
		web_tool_kwargs={
        	"browser_config" : {
            	"headless": False,
                "new_context_config": BrowserContextConfig(
                	save_downloads_path= os.path.join(os.path.dirname(__file__), "../tempConversation/tempDownloads")
            	)
			}
    	}
	)

	# Tool self.registration required in GroupChat context for proper function map entry
	# Allows agent to both propose tool use and execute that tool on proposition
	for tool in agent.tools:
		tool.register_for_llm(agent)
		tool.register_for_execution(agent)

	# Hook the document sorting function
	agent.register_hook("process_message_before_send", sortDocuments)

	return agent
