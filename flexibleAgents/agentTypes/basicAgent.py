from typing import Dict, List, Literal
from autogen import ConversableAgent
from pydantic import BaseModel



# Define reasoning agent response format
class basicAgentResponse(BaseModel):
	message: str						# Detailed scientific explanation of the topic


def basicAgent(llm_config, name = "BasicAgent") -> ConversableAgent:
	systemMessage = ""

	description = """
        The BASIC AGENT is a simple wrapper around a backbone LLM without any additional system message instructions or agentic capabilities.
	"""

	basic_llm_config = llm_config.copy()
	basic_llm_config["response_format"] = basicAgentResponse
	basic_llm_config["temperature"] = 0.01

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = basic_llm_config,
		human_input_mode="NEVER"
	)