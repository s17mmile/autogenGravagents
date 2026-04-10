from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel



# Define coding agent response format
class codingAgentResponse(BaseModel):
	message: str						# Message giving an overview of what has been coded
	codeSnippet: str					# Code snippet generated



def codingAgent(chat, name = "CodingAgent") -> ConversableAgent:
	systemMessage = f"""
		You are a Coding Agent whose purpose is to write python code based on given instructions.
		You have the ability to generate a code snippet that can be used by other agents or executed in a local environment.
		If you are asked to write multiple code snippets, you should combine them into a single, coherent code snippet that can be executed independently OR separate the code snippets, providing feedback that you can only generate one at a time.

		Notes:
		Unless explicitly asked, avoid taking user input in your code snippets, as they should be automatically executable with preset constants.
		Never include unicode characters in your code snippets, as they may cause issues when executing the code. If you want to include special characters, use a corresponding spelled-out version.
		When generating a plot with your code, only save it - do not attempt to display it with plt.show() or similar, as the code will be executed in a non-interactive environment where displaying plots is not possible.

		Your output includes a message field and a codeSnippet field:
		- The message field should give a quick overview of the code generated and its purpose. Include relevant information about how it addresses the given instructions and - if any - what libraries are used and how.
		- The codeSnippet field should contain the generated code snippet as a single string representing a valid piece of python code that can be executed independently.
	"""

	description = """
		The CODING AGENT is responsible for writing single python code snippets (one at a time!) based on given instructions.
	"""

	coding_llm_config = chat.llm_config.copy()
	coding_llm_config["response_format"] = codingAgentResponse

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = coding_llm_config,
		human_input_mode="NEVER"
	)
