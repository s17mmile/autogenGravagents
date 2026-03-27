import os, codecs
from typing import Dict, List
from anyio import Path

from autogen.agents.experimental import DocAgent
from autogen.agents.experimental import VectorChromaQueryEngine
from pydantic import BaseModel

# Needed for vector DB management
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Note: the documentRetrievalAgent also has browser-use in the background, which causes a few warnings internally:
# - a pydantic warning about an improper field
# - a future warning about google genai. 
# - a warning about the "stream" parameter
# - a warning about pkg_resources deprecation
# This is out of my control and does not affect execution.
# I believe this occurs because ag2 requires on old version of the browser-use tool (0.1.37). I will try updating it to see if it fixes warnings, but no promises.

# Define doc agent response format
class documentRetrievalAgentResponse(BaseModel):
	message: str								# Answer to the query based on retrieved documents
	retrievedDocumentNames: List[str]			# List of names of retrieved documents (so the human can cross-check sources)

# The Document Retrieval Agent is responsible for retrieving relevant documents from a local document corpus to answer queries posed by other agents.
# It should utilize document search to back up argumentations or answer questions with facts from given sources.
# Its knowledge is based solely on the documents in the provided corpus, which can be expanded upon through several runs of this system.
# It uses a chroma vector database (stored alongside the document corpus) to store and query the ingested documents, and will only retrieve documents from the provided corpus to ensure verifiable and accurate information retrieval.
def documentRetrievalAgent(llm_config, name = "DocumentRetrievalAgent") -> DocAgent:
	# Define path for document corpus and create folder if it doesn't exist
	corpusPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentCorpus"))
	os.makedirs(corpusPath, exist_ok=True)

	# Parsed Documents Path (for chromaDB ingestion)
	parsedDocsPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "parsedDocs"))
	os.makedirs(parsedDocsPath, exist_ok=True)

	# Create Chroma client and point it to persistent collection in the DB
	chromaDbPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chromaDb"))

	# Build an embedding function which uses the local GPT-4o API
	embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
		api_key=llm_config["config_list"][0]["api_key"],         	# same key as for llm_config (extracted from first config list entry)
		api_base=llm_config["config_list"][0]["base_url"],      	# same base_url as llm_config (extracted from first config list entry)
		model_name="text-embedding-3-small"							# A cheap embedding function - could also use others
	)

	# Build AG2 Query Engine using the chroma collection and the same llm config as for the agent backbone
	# The LLM here is used to handle the DB queries (separate from the rest of the docAgent!)
	query_engine = VectorChromaQueryEngine(
		db_path=chromaDbPath,
		embedding_function=embedding_fn,
		llm=llm_config,
		collection_name="memoryBank"
	)

	systemMessage = f"""
		You are a DOCUMENT RETRIEVAL AGENT specializing in answering queries based on retrieved documents.
		All local documents MUST be ingested from {corpusPath}.
		
		Your responsibilities:
		1. Understand the specific question or statement presented to you.
		2. Ingest ALL documents provided in the documentCorpus and quote them to answer the question or respond to the statement.

		Your output includes a message field and a retrievedDocumentNames field:
		- The message field should contain your answer to the query based on the information from the retrieved documents.
		- The retrievedDocumentNames field should list the names of the documents you retrieved to support your answer.
	"""

	description = """
		The DOCUMENT RETRIEVAL AGENT is responsible for retrieving relevant documents from a local document corpus to answer queries posed by other agents.
		It should utilize document search to back up argumentations or answer questions with facts from given sources.
	"""

	documentRetrieval_llm_config = llm_config.copy()
	documentRetrieval_llm_config["response_format"] = documentRetrievalAgentResponse
	documentRetrieval_llm_config["temperature"] = 0.1

	# Using a given collection name is needed to retain knowledge across runs
	doc_agent = DocAgent(
		name=name,
		system_message=systemMessage,
		llm_config=documentRetrieval_llm_config,
		parsed_docs_path=parsedDocsPath,
		query_engine=query_engine
	)

	response = doc_agent.run(
		message = f"Ingest all of the following: {[os.path.abspath(os.path.join(corpusPath, doc)) for doc in os.listdir(corpusPath)]}",
		max_turns=1,
		silent=False
	)

	response.process()

	quit()

	return doc_agent
