from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel

# Define query agent response format
class queryAgentResponse(BaseModel):
	message: str					# Message to the user about the task breakdown and distribution
	nextAgentName: str				# Name of the next agent to speak
	subtasks: list[str]				# List of sub-tasks

# Query Agent takes in the initial query and then delegates tasks to other agents
# Based on the user's input, it breaks down the task into sub-tasks for other agents to handle
# Each Conversation Config must include a Query Agent as the starting point
def queryAgent(llm_config, name = "QueryAgent", allowedTransitions: List[str] = []) -> ConversableAgent:
	systemMessage = f"""
		You are a QUERY AGENT specializing in breaking down user requests into manageable sub-tasks for a team of specialized agents.
		You will receive the user's query as well as a short description of the agent conversation configuration.
		Based on this query, you should check the capabilities of the available agents and break down the task into manageable sub-tasks to be handled by other agents.

		Your responsibilities:
		1. Understand the user's overall goal from their input.
		2. Decompose the goal into specific sub-tasks that can then be assigned to specialized agents (e.g., coding agent, interpreter agent) by the group chat manager.
		
		Your output includes a message field a nextAgentName field, and a subtasks field:
		- The message field should include your understanding of the tasks and suggested next steps towards solving it. If requirements are not given, this should be included here.
		- The nextAgentName field should include the name of another agent in the agentic system. It must strictly be one of the following names: {allowedTransitions}.
		Only return an emtpy field (terminating conversation) if no transition is allowed!
	"""

	print(name, allowedTransitions)

	description = """
		The QUERY AGENT is responsible for breaking down user requests into manageable sub-tasks based on the capabilities of the available agents in the conversation configuration.
		It ensures that the task flow is supported by the allowed agent transitions and leverages a Human Agent if needed for clarification.
	"""

	query_llm_config = llm_config.copy()
	query_llm_config["response_format"] = queryAgentResponse
	query_llm_config["temperature"] = 0.1

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = query_llm_config,
		human_input_mode="NEVER"
	)
