import os
from autogen.agents.experimental import WebSurferAgent
from pydantic import BaseModel
from typing import Dict, List

# Define research agent response format
class researchAgentResponse(BaseModel):
	message: str								# Overview of found documents/visited web pages
	foundDocumentNames: List[str]			    # List of names of documents (so the human can cross-check sources)

# TODO Create Agent tool to save documents to local folder? Or is that already possible?

# The Research Agent is responsible for conducting research on the web to find relevant information and sources to answer queries posed by other agents.
# It is responsible for finding and downloading relevant documents from the web, which can then be ingested into the system and used by the Document Retrieval Agent to answer questions with verifiable sources.
def researchAgent(llm_config, name = "ResearchAgent") -> WebSurferAgent:
	# Define path for document corpus
	documentCorpusPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentCorpus"))

	systemMessage = f"""
		You are a RESEARCH AGENT specializing in finding and downloading relevant documents from the web.
		You will receive queries from other agents in the system.
		Based on the task and the received question/statement, you should find and download relevant documents (HTML, Text Files, PDFs, etc.) from the web that can be ingested into the system.
		These documents are to be stored in the local document corpus directory {documentCorpusPath}.
		After downloading, these documents should be available for ingestion by the Document Retrieval Agent to answer questions with verifiable sources.
		
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
	research_llm_config["temperature"] = 0

	# Using a given collection name is needed to retain knowledge across runs
	return WebSurferAgent(
		name = name,
		system_message = systemMessage,
		llm_config = research_llm_config,
		web_tool="browser_use"
	)
