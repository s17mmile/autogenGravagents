# Set UTF-8 encoding for stdout to ensure proper handling of special characters in document retrieval and processing
import sys
import os

# Typing imports
from typing import Dict, List
from pydantic import BaseModel
from pathlib import Path
import PyPDF2

# Query engine, LLM and AG2 imports
print("Importing VectorChromaQueryEngine and related...")
from llama_index.llms.openai_like import OpenAILike
from autogen.agents.experimental import VectorChromaQueryEngine
from autogen.agents.experimental.document_agent.chroma_query_engine import VectorChromaCitationQueryEngine

print("Importing AG2 DocAgent...")
from autogen.agents.experimental import DocAgent

# Docling imports for fine-grained document parsing control before ingestion
print("Importing Docling...")
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Note: the documentRetrievalAgent also has browser-use in the background, which causes a few warnings internally:
# - a pydantic warning about an improper field
# - a future warning about google genai. 
# - a warning about the "stream" parameter
# - a warning about pkg_resources deprecation
# This is out of my control and does not affect execution.
# I believe this occurs because ag2 requires on old version of the browser-use tool (0.1.37). I will try updating it to see if it fixes warnings, but no promises.

def buildQueryEngine(llmconfig, chromaDbPath, collection_name):
	# Define LLM instance for query engine (doesn't just use llmconfig dict, client must be instantiated manually)
	queryEngineLLM = OpenAILike(
		model=llmconfig.get("model"),
		api_base=llmconfig.get("base_url", None),
		api_key=llmconfig.get("api_key", None),
		temperature=llmconfig.get("temperature", 0.1)
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

# Ingest all not-yet ingested PDF documents from the corpus (if any)
def ingestNewPDFs(doc_agent, corpusPath, parsedDocsPath):
	corpusPathObject = Path(corpusPath)
	parsedDocsPathObject = Path(parsedDocsPath)
	
	pdf_pipeline_options = PdfPipelineOptions(
		generate_page_images=False,
		generate_picture_images=False,
		do_table_structure=True,
		do_ocr=True,
		generate_parsed_pages=False,
		page_batch_size=1,  # Critical for OOM
		ocr_batch_size=1,
		layout_batch_size=1,
		document_timeout=180.0,
	)
	
	converter = DocumentConverter(
		format_options={
			InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
		}
	)
	
	generatedPaths = []
	
	for pdf_path in corpusPathObject.glob("*.pdf"):
		md_path = parsedDocsPathObject / f"{pdf_path.stem}.md"
		
		if md_path.exists():
			continue
		
		# Split large PDF into 10-page chunks
		chunk_size = 10
		with open(pdf_path, "rb") as f:
			reader = PyPDF2.PdfReader(f)
			total_pages = len(reader.pages)
			
			chunk_mds = []
			for start in range(0, total_pages, chunk_size):
				end = min(start + chunk_size, total_pages)
				
				# Create chunk PDF
				chunk_pdf_path = parsedDocsPathObject / f"{pdf_path.stem}_chunk_{start+1:03d}-{end:03d}.pdf"
				writer = PyPDF2.PdfWriter()
				for i in range(start, end):
					writer.add_page(reader.pages[i])
				with open(chunk_pdf_path, "wb") as chunk_f:
					writer.write(chunk_f)
				
				# Convert chunk
				try:
					result = converter.convert(str(chunk_pdf_path))
					chunk_md = result.document.export_to_markdown()
					chunk_mds.append(chunk_md)
				finally:
					chunk_pdf_path.unlink()  # Cleanup
				
			# Combine chunks into final MD
			full_md = "\n\n---\n\n".join(chunk_mds)
			md_path.write_text(full_md, encoding="utf-8")
		
			generatedPaths.append(str(md_path))
	
	doclist = ["A-Level-Chemistry.md", "cambridge international as and a level physics coursebook - public.md", "Pure Mathematics Textbook.md"]

	if len(generatedPaths) > 0:
		# Ingest all the newly parsed documents
		response = doc_agent.run(
			message=f"Ingest all of the following: {generatedPaths}, then terminate immediately.",
			max_turns=1,
			silent=False
		)
		response.process()
	
	return

# Define doc agent response format
class documentRetrievalAgentResponse(BaseModel):
	message: str								# Answer to the query based on retrieved documents
	retrievedDocumentNames: List[str]			# List of names of retrieved documents (so the human can cross-check sources)

# The Document Retrieval Agent is responsible for retrieving relevant documents from a local document corpus to answer queries posed by other agents.
# It should utilize document search to back up argumentations or answer questions with facts from given sources.
# Its knowledge is based solely on the documents in the provided corpus, which can be expanded upon through several runs of this system.
# It uses a chroma vector database (stored alongside the document corpus) to store and query the ingested documents, and will only retrieve documents from the provided corpus to ensure verifiable and accurate information retrieval.
def documentRetrievalAgent(chat, name = "DocumentRetrievalAgent") -> DocAgent:
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
			--> If you do not have the answer to a query, explicitly state that you should not be req-queried for the same information.
		- The retrievedDocumentNames field should list the names of the documents you retrieved to support your answer.
	
		If any agent asks for clarification on how to use a specific library function or API, you should answer as comprehensively as possible:
		- search for or create code examples of that function or API as a reference.
		- Include information about import statements, function signatures and syntax, argument formats, and example code snippets as needed.
		- Comment on any known pitfalls or other things to watch out for when using that function or API, based on the information in the documents you have ingested.
			--> If you do not have any documentation about a specific library function or API in your current knowledge base, simply state that these documents are missing and ask for human input. Do not attempt to fill in the gaps yourself, as you may provide inaccurate information. Instead, suggest that the agent asking for this information should query you for relevant code examples or documentation on that function or API.
	"""

	description = f"""
		The DOCUMENT RETRIEVAL AGENT is responsible for retrieving relevant documents from a local document corpus or the web to answer queries posed by other agents.
		It should utilize document search to back up argumentations or answer questions with facts from given sources.
		It can be given natural language queries including data ingestion requests from given URLs or local files.
		You may aks this agent for code examples to be used as a reference for implementation of library-specific functions or APIs. 
	"""

	documentRetrieval_llm_config = chat.llm_config.copy()

	# Build query engine for the DocAgent to use (also uses gpt-4o-mini)
	query_engine = buildQueryEngine(llmconfig=documentRetrieval_llm_config, chromaDbPath=chromaDbPath, collection_name=collection_name)

	documentRetrieval_llm_config["response_format"] = documentRetrievalAgentResponse

	# Using a given collection name is needed to retain knowledge across runs
	doc_agent = DocAgent(
		name=name,
		system_message=systemMessage,
		llm_config=documentRetrieval_llm_config,
		parsed_docs_path=parsedDocsPath,
		collection_name=collection_name,
		query_engine=query_engine
	)

	ingestNewPDFs(doc_agent, corpusPath, parsedDocsPath)

	return doc_agent
