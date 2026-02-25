import os
from typing import Dict, List
print("Importing DocAgent")
from autogen.agents.experimental import DocAgent
print("Importing VectorChromaQueryEngine")
from autogen.agents.experimental import VectorChromaQueryEngine
print("Done with Large imports")
from pydantic import BaseModel

# Needed for vector DB management
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Define fact checking agent response format
class documentRetrievalAgentResponse(BaseModel):
	message: str								# Answer to the query based on retrieved documents
	retrievedDocumentNames: List[str]			# List of names of retrieved documents (so the human can cross-check sources)

# The Document Retrieval Agent is responsible for retrieving relevant documents from a local document corpus to answer queries posed by other agents.
# It should utilize document search to back up argumentations or answer questions with facts from given sources.
# Its knowledge is based solely on the documents in the provided corpus, which can be expanded upon through several runs of this system.
# It uses a chroma vector database (stored alongside the document corpus) to store and query the ingested documents, and will only retrieve documents from the provided corpus to ensure verifiable and accurate information retrieval.
def documentRetrievalAgent(llm_config, name = "DocumentRetrievalAgent") -> DocAgent:
	# Define path for document corpus
	originalDocPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentCorpus"))

	# Create Chroma client and point it to persistent collection in the DB
	chromaDbPath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chromaDb"))
	chroma_client = chromadb.PersistentClient(path=chromaDbPath)

	# Build an embedding function which uses the local GPT-4o API
	embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
		api_key=llm_config["config_list"][0]["api_key"],         	# same key as for llm_config (extracted from first config list entry)
		api_base=llm_config["config_list"][0]["base_url"],      	# same base_url as llm_config (extracted from first config list entry)
		model_name="text-embedding-3-small"							# TODO Check if this is the correct embedding function
	)

	collection = chroma_client.get_or_create_collection(
		name="memoryBank",
		embedding_function=embedding_fn,
	)

	# Build AG2 Query Engine using the chroma collection and the same llm config as for the agent backbone
	# The LLM here is used to handle the DB queries (separate from the rest of the docAgent!)
	query_engine = VectorChromaQueryEngine(
		collection=collection,
		llm=llm_config,
	)

	systemMessage = f"""
		You are a DOCUMENT RETRIEVAL AGENT specializing in answering queries based on retrieved documents.
		You will receive questions or statements from other agents in the system.
		Based on the task and the received question/statement, you should retrieve relevant documents from the available corpus to provide a verifiable answer and back up claims.
		All of the documents you retrieve should be from the provided corpus; they are locally stored in the directory {originalDocPath}.
		These documents are to be ingested into a vector database that you can query to find relevant information.
		Make absolutely sure to only use documents from this corpus to answer the queries - if no relevant documents are found, state that you could not find any relevant information in the corpus.
		Never retrieve documents that are not part of the provided corpus.
		
		Your responsibilities:
		1. Understand the specific question or statement presented to you.
		2. Retrieve relevant documents from the available corpus that can help answer the question or support the statement.
		3. Provide a concise answer to the question or statement based on the retrieved documents, referencing each document by name where needed.
		4. List the names of the documents you retrieved to support your answer. This allows for transparency and human verification of your sources if needed.

		Your output includes a message field and a retrievedDocumentNames field:
		- The message field should contain your answer to the query based on the information from the retrieved documents.
		- The retrievedDocumentNames field should list the names of the documents you retrieved to support your answer.
	"""

	description = """
		The DOCUMENT RETRIEVAL AGENT is responsible for retrieving relevant documents from a local document corpus to answer queries posed by other agents.
		It should utilize document search to back up argumentations or answer questions with facts from given sources.
		Its knowledge is based solely on the documents in the provided corpus, which can be expanded upon through several runs of this system.
	"""

	documentRetrieval_llm_config = llm_config.copy()
	documentRetrieval_llm_config["response_format"] = documentRetrievalAgentResponse
	documentRetrieval_llm_config["temperature"] = 0.05

	# Using a given collection name is needed to retain knowledge across runs
	return DocAgent(
		name = name,
		system_message = systemMessage,
		llm_config = documentRetrieval_llm_config,
		query_engine=query_engine
	)
