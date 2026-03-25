import json
from typing import Any, Dict, List, Union
from autogen import ConversableAgent
from pydantic import BaseModel

# Basic Human Agent
def humanAgent(llm_config, name = "Human") -> ConversableAgent:
	description = """
		The HUMAN AGENT serves as an interface for direct human input within the agent conversation system.
		It is designed to receive queries or tasks from other agents and provide human responses when necessary.
	"""

	agent = ConversableAgent(
		name = name,
		description=description,
		human_input_mode="ALWAYS"
	)

	return agent
