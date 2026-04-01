import os
import shutil
from autogen import ConversableAgent
from browser_use import BrowserContextConfig
from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Union

from autogen.agents.experimental import WebSurferAgent

# Note: the documentCollectionAgent is based on browser-use, which causes a few warnings internally:
# - a pydantic warning about an improper field
# - a future warning about google genai. 
# - a warning about the "stream" parameter
# - a warning about pkg_resources deprecation
# This is out of my control and does not affect execution.
# I believe this occurs because ag2 requires on old version of the browser-use tool (0.1.37). I will try updating it to see if it fixes warnings, but no promises.

# Define document collection agent response format
class documentCollectionAgentResponse(BaseModel):
	message: str									# Overview of found documents/visited web pages, important takeaways
	relevantURLs: List[str]							# List of relevant URLs



# The Document Collection Agent is responsible for collecting relevant documents from the web.
def documentCollectionAgent(llm_config, name = "DocumentCollectionAgent") -> WebSurferAgent:
	systemMessage = f"""
		You are a DOCUMENT COLLECTION AGENT specializing in finding documents or data from the web.
		You will find information on the internet and save  URLs to documents or web pages that can be used for Retrieval Augmented Generation.
		When searching for information, use the most reputable sources possible, such as academic papers, official websites, and well-known news outlets. Avoid using information from unreliable sources or forums if possible.  
		
		Your responsibilities:
		1. Search the web for relevant information based on the queries you receive.
        2. Save URLs to documents or web pages that can be used for Retrieval Augmented Generation.
		3. Respond to queries with short summaries of what you found in your search and the URLs of relevant documents or web pages.
		4. Do not download the documents yourself, just provide the URLs. The Document Retrieval Agent will handle downloading and ingesting documents based on the URLs you provide.
		
		Your output includes a message field and a relevantURLs field:
		- The message field should contain an overview of the pages visited and important takeaways to answer the query.
		- The relevantURLs field should list the URLs of the documents or web pages you found.
	"""

	description = """
		The DOCUMENT COLLECTION AGENT is responsible for finding URLs of relevant documents from the web to be ingested into a retrieval augmented generation system.
		An RAG system should attempt to download and insget information from these URLs.
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
			}
    	}
	)

	# Tool self-registration required in GroupChat context for proper function map entry
	# Allows agent to both propose tool use and execute that tool on proposition
	# The tool, in this case, is browser-use: without this, no browser can actually be used.
	for tool in agent.tools:
		tool.register_for_llm(agent)
		tool.register_for_execution(agent)

	return agent
