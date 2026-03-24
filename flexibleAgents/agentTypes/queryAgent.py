from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel

# Define query agent response format
class queryAgentResponse(BaseModel):
	message: str					# Message to the user about the task breakdown and distribution
	subtasks: List[str]				# List of sub-tasks
	isTaskComplete: bool				# Boolean whether to terminate the chat or not.

# Query Agent takes in the initial query and then delegates tasks to other agents
# Based on the user's input, it breaks down the task into sub-tasks for other agents to handle
# Each Conversation Config must include a Query Agent as the starting point
def queryAgent(llm_config, name = "QueryAgent") -> ConversableAgent:
	systemMessage = f"""
		You are a QUERY AGENT specializing in breaking down user requests into manageable sub-tasks for a team of specialized agents.
		You will receive the user's query as well as a short description of the agent conversation configuration.
		Based on this query, you should check the capabilities of the available agents and break down the task into manageable sub-tasks to be handled by other agents.
		Also, you are responsible for stopping a conversation once you believe the other agents have successfully finished each subtask.

		Your responsibilities:
		1. Understand the user's overall goal from their input.
		2. Decompose the goal into specific sub-tasks that can then be assigned to specialized agents (e.g., coding agent, interpreter agent) by the group chat manager.
		3. Interpret whether or not the work done by other agents is sufficient for the overall task at hand. If you believe the question posed has been properly answered, quit the conversation.
		
		Your output includes a message field, a subtasks field, and a isTaskComplete field:
		- The message field should include your understanding of the tasks and suggested next steps towards solving it. If requirements are not given, this should be included here.
		- The subtasks field should contain a breakdown of the overall task into simpler-to-manage tasks for other agents.
		- The isTaskComplete field contains a boolean value. Set this to one if and only if the agentic system has sufficiently answered the original query. If not, leave it at zero.
	"""

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
