from typing import Dict, List
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pydantic import BaseModel



# Define execution agent response format
class executionAgentResponse(BaseModel):
	message: str						# Message giving an overview of execution results
    createdFiles: List[str]			    # List of created files



def executionAgent(llm_config, name = "ExecutionAgent") -> ConversableAgent:
	systemMessage = f"""
		You are an Execution Agent whose purpose is to execute code in a local command line environment.
        You have access to a local command line code executor that can run code and return the results.
        When you receive code to execute from another agent, use the code executor to run it.
	"""

	description = """
		The EXECUTION AGENT is responsible for executing code in a local command line environment.
	"""

	execution_llm_config = llm_config.copy()
	execution_llm_config["response_format"] = executionAgentResponse
	execution_llm_config["temperature"] = 0

    executor = LocalCommandLineCodeExecutor(
        timeout=20,                             # Timeout for each code execution in seconds.
        work_dir="tempConversation",            # Use the temporary conversation directory as the working directory.
    )

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = execution_llm_config,
		human_input_mode="NEVER"
	)
