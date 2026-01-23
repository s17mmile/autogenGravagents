from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel

# Basic Human Agent
def humanAgent(llm_config, name = "Human", allowedTransitions: List[str] = []) -> ConversableAgent:
	description = """
		The HUMAN AGENT serves as an interface for direct human input within the agent conversation system.
		It is designed to receive queries or tasks from other agents and provide human responses when necessary.
	"""

	return ConversableAgent(
		name = name,
		description=description,
		human_input_mode="ALWAYS"
	)
