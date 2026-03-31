# Set UTF-8 encoding for stdout to ensure proper handling of special characters in document retrieval and processing
import sys
import os

# Typing imports
from typing import Dict, List
from anyio import Path
from pydantic import BaseModel

# Query engine, LLM and AG2 imports
from llama_index.llms.openai import OpenAI
from autogen.agents.experimental import VectorChromaQueryEngine
from autogen.agents.experimental.document_agent.chroma_query_engine import VectorChromaCitationQueryEngine
from autogen.agents.experimental import DocAgent

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

def buildQueryEngine(llm_config, chromaDbPath, collection_name):
	# Define LLM instance for query engine --> hardcoded to be OpenAI here because it's a proof of concept and proper LLMConfig usage is annoying.
	queryEngineLLM = OpenAI(
						temperature=llm_config["temperature"],
						model=llm_config["model"],
						api_key=llm_config["api_key"],
						base_url=llm_config.get("base_url", None)
						)

	# Build AG2 Query Engine using the chroma collection and the same llm config as for the agent backbone
	# The LLM here is used to handle the DB queries (separate from the rest of the docAgent!)
	# Currently uses default embedding which uses the regular OpenAI API!
	# --> Would need to customize this to use a local embedding model to avoid commercial API calls for the vector DB management. It's just a little annoying, as that includes creating an OpenAI instance instead of using the regular LLMconfig.
	# Optional use of citation query engine --> didn't really seem to be that much more powerful.
	query_engine = VectorChromaQueryEngine(
		db_path=chromaDbPath,
		collection_name=collection_name,
		# enable_query_citations=True,
		llm=queryEngineLLM
	)

	return query_engine


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
	collection_name="persistentMemoryBank"

	# Create Chroma client and point it to persistent collection in the DB
	chromaDbPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chromaDatabase"))

	systemMessage = f"""
		You are a DOCUMENT RETRIEVAL AGENT specializing in answering queries based on retrieved documents.
		All local documents MUST be ingested from {corpusPath}.
		
		Your responsibilities:
		1. Understand the specific question or statement presented to you.
		2. Ingest ALL documents provided in the documentCorpus and quote them to answer the question or respond to the statement.
		3. If provided links, download the web page or document linked and ingest it.

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
	documentRetrieval_llm_config["temperature"] = 0.0

	# Build query engine for the DocAgent to use
	query_engine = buildQueryEngine(llm_config=documentRetrieval_llm_config, chromaDbPath=chromaDbPath, collection_name=collection_name)

	# Using a given collection name is needed to retain knowledge across runs
	doc_agent = DocAgent(
		name=name,
		system_message=systemMessage,
		llm_config=documentRetrieval_llm_config,
		parsed_docs_path=parsedDocsPath,
		collection_name=collection_name,
		query_engine=query_engine
	)

	# Ingest all not.yet ingested documents from the corpus (if any)
	docsToParse = [os.path.abspath(os.path.join(corpusPath, doc)) for doc in os.listdir(corpusPath) if not os.path.isfile(os.path.splitext(os.path.join(parsedDocsPath, doc))[0]+".md")]
	if docsToParse:
		print(f"Document Retrieval Agent: Ingesting documents from corpus for the first time: {docsToParse}")
		response = doc_agent.run(
			message = f"Ingest all of the following: {docsToParse}, then terminate immediately.",
			max_turns=1,
			silent=False
		)

		response.process()

	return doc_agent
