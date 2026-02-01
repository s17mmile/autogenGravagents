from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel



# Define coding agent response format
class codingAgentResponse(BaseModel):
	message: str						# Message giving an overview of what has been coded
	codeSnippets: List[str]				# List of code snippets generated



def codingAgent(llm_config, name = "CodingAgent") -> ConversableAgent:
	systemMessage = f"""
		You are a Coding Agent whose purpose is to write python code based on given instructions.
		You have the ability to generate code snippets that can be used by other agents or executed in a local environment.

		Your output includes a message field and a codeSnippets field:
		- The message field should give a quick overview of the code generated and its purpose. Include relevant information about how it addresses the given instructions and - if any - what libraries are used and how.
		- The codeSnippets field should separately list any code snippets that were generated during the process. Each code snippet should be a single string representing a valid piece of python code that can be executed independently.
	"""

	description = """
		The CODING AGENT is responsible for writing python code snippets based on given instructions.
	"""

	coding_llm_config = llm_config.copy()
	coding_llm_config["response_format"] = codingAgentResponse
	coding_llm_config["temperature"] = 0

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = coding_llm_config,
		human_input_mode="NEVER"
	)
