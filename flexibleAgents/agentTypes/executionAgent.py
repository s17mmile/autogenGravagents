import os
from typing import Dict, List
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pydantic import BaseModel



# Define execution agent response format
class executionAgentResponse(BaseModel):
	message: str						# Message giving an overview of execution results
	result: str
	createdFiles: List[str]				# List of files created during execution



def executionAgent(llm_config, name = "ExecutionAgent") -> ConversableAgent:
	systemMessage = f"""
		You are an Execution Agent whose purpose is to execute code in a local command line environment.
		You have access to a local command line code executor that can run code and return the results.
		When you receive code to execute from another agent, use the code executor to run it.

		Your output includes a message field and a createdFiles field:
		- The message field should give a quick overview of the code executed, whether or not the execution was successful and why. If there were errors, provide a brief explanation.
		- The result field should contain the output or result of the code execution if it can be represented as a short string or number.
		- The createdFiles field should list any files that were created during the execution process, such as images or text outputs.
	"""

	description = """
		The EXECUTION AGENT is responsible for executing code snippets from other agents' responses in a local command line environment.
	"""

	execution_llm_config = llm_config.copy()
	execution_llm_config["response_format"] = executionAgentResponse
	execution_llm_config["temperature"] = 0

	executor = LocalCommandLineCodeExecutor(
		timeout=20,							 # Timeout for each code execution in seconds.
		work_dir=f"{os.path.dirname(__file__)}/../tempConversation",			# Use the temporary conversation directory as the working directory.
	)

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = execution_llm_config,
		code_execution_config={"executor": executor},
		human_input_mode="NEVER"
	)
