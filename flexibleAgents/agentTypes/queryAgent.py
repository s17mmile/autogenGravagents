from typing import Dict
from autogen import ConversableAgent
from pydantic import BaseModel

# Define query agent response format
class queryAgentResponse(BaseModel):
	isAgentConfigValid: bool		# Whether the current agent configuration supports the user's request
	recommendedAgentConfig: str		# Suggestions for different agent config if current config is insufficient
	nextAgentName: str				# Name of the next agent to speak
	subtasks: Dict[str, str]		# Mapping of agent names to their assigned sub-tasks
	messageToUser: str				# Message to the user about the task breakdown and distribution

# Query Agent takes in the initial query and then delegates tasks to other agents
# Based on the user's input, it breaks down the task into sub-tasks for other agents to handle
# Each Conversation Config must include a Query Agent as the starting point
def queryAgent(name = "QueryAgent", llm_config) -> ConversableAgent:
	systemMessage = """
		You are a QUERY AGENT specializing in breaking down user requests into manageable sub-tasks for a team of specialized agents.
		You will receive only the user's query.
		Based on this query, you should check the capabilities of the available agents and break down the task into manageable sub-tasks to be handled by other agents.

		Your responsibilities:
		1. Understand the user's overall goal from their input.
		2. Decompose the goal into specific sub-tasks that can then be assigned to specialized agents (e.g., coding agent, interpreter agent) by the group chat manager.
		3. Check if the needed agents are available in the conversation configuration. If not, inform the user that their request requires different agents.
		4. Check that the allowed agent transitions support the planned task flow. If not, inform the user that their request requires a different configuration.

		Should you or another agent in the system require human input, you may call upon a Human Agent to assist with gathering information or clarifying requirements.
	"""

	query_llm_config = llm_config.copy()
	query_llm_config["response_model"] = queryAgentResponse
	query_llm_config["temperature"] = 0.1

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		llm_config = query_llm_config,
		human_input_mode="NEVER"
	)
