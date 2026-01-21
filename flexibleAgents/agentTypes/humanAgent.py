from typing import Dict
from autogen import ConversableAgent
from pydantic import BaseModel

# Basic Human Agent
def humanAgent(llm_config, name = "Human") -> ConversableAgent:
	return ConversableAgent(
		name = name,
		human_input_mode="ALWAYS"
	)
