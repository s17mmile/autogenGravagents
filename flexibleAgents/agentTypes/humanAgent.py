from typing import Dict
from autogen import ConversableAgent
from pydantic import BaseModel

# Basic Human Agent
def humanAgent(name = "Human", llm_config = None) -> ConversableAgent:
	return ConversableAgent(
		name = name,
		human_input_mode="ALWAYS"
	)
