import os, json
from typing import Any, Dict, List
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pydantic import BaseModel
from pathlib import Path


# Define execution agent response format
class executionAgentResponse(BaseModel):
	message: str						# Message giving an overview of execution results
	result: str							# Result or output of the code execution
	createdFiles: List[str]				# List of files created during execution
	requiredDependencies: List[str]		# List of any required, but not yet dependencies for the executed code

def injectSnippets(agent, messages, sender, config):
	# Extract code snippets as single strings from incoming message
	lastMessage = json.loads(messages[-1].get("content")) if "content" in messages[-1] else {}
	snippet = lastMessage["codeSnippet"] if "codeSnippet" in lastMessage else []
	
	if not snippet:
		return False, {}
	
	# Inject as single "code prompt" message --> Yes, this loses the format going in, but that should be fine.
	code_content = f"```python\n{snippet}\n```"
	messages.append({
		"content": f"Execute this code snippet:\n{code_content}",
		"role": "user"
	})
	
	return False, {}

# Update code executor's working directory to the current conversation path before each execution, to ensure that any files created are stored in the proper directory.
# Hooked to run before reply
def update_working_directory(
	agent: ConversableAgent,
	messages: list[dict[str, Any]]
) -> None:
    executor = agent.code_executor
    if executor is not None:
        new_dir = Path(agent.chat.getConversationPath())
        os.makedirs(new_dir, exist_ok=True)
		# This is meant to be read-only, so we need to directly access internals. Little unclean, but eh.
        executor._work_dir = new_dir
    return messages

	

def executionAgent(chat, name = "ExecutionAgent") -> ConversableAgent:
	systemMessage = f"""
		You are an Execution Agent whose purpose is to execute code in a local command line environment.
		You have access to a local command line code executor that can run code and return the results.
		When you receive code to execute from another agent, use the code executor to run it.

		Your output includes a message field and a createdFiles field:
		- The message field should give a quick overview of the code executed, whether or not the execution was successful and why. If there were errors, provide a brief explanation.
		- The result field should contain the output or result of the code execution if it can be represented as a short string or number.
		- The createdFiles field should list any files that were created during the execution process, such as images or text outputs. It should also include the names of the scripts created during execution.
		- The requiredDependencies field should list any required, but not yet installed dependencies needed to run the code. If dependencies are missing, hand control back to the human agent to install these dependencies.
	"""

	description = """
		The EXECUTION AGENT is responsible for executing code snippets from other agents' responses in a local command line environment.
	"""

	execution_llm_config = chat.llm_config.copy()
	execution_llm_config["response_format"] = executionAgentResponse
	execution_llm_config["temperature"] = 0

	executor = LocalCommandLineCodeExecutor(
		timeout=60,							   	# Timeout for each code execution in seconds.
		work_dir=chat.getConversationPath(),	# Use the temporary conversation directory as the working directory.
	)

	agent = ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = execution_llm_config,
		code_execution_config={"executor": executor},
		human_input_mode="NEVER"
	)

	# Registering a new attribut to this agent to keep track of the chat instance.
	# THis is necessary for the code executor to have access to the conversation path for code execution, as it can be changed and needs to be re-fetched.
	agent.chat = chat

	# To enable code execution from pydantic response formats, we need to pull code snippets from their field and inject them into message history. 
	agent.register_reply(
		trigger = [ConversableAgent, None],
		reply_func = injectSnippets,
		position = 1
	)

	# Hook to update working directory before reply generation
	agent.register_hook("update_agent_state", update_working_directory)

	return agent
