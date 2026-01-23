from typing import Dict, List, Literal
from autogen import ConversableAgent
from pydantic import BaseModel



# Define reasoning agent response format
class reasoningAgentResponse(BaseModel):
	message: str						# Detailed scientific explanation of the topic
	nextAgentName: str					# Name of the next agent to speak



def reasoningAgent(llm_config, name = "ReasoningAgent", allowedTransitions: List[str] = []) -> ConversableAgent:
	systemMessage = f"""
		You are a REASONING AGENT specializing in providing detailed scientific explanations and relevant facts.
		You will receive specified tasks from a query agent or other agents in the system.
		Based on the task, you should provide a thorough scientific explanation.

		Your responsibilities:
		1. Understand the specific scientific topic or question presented to you.
		2. Provide a detailed explanation that is accurate and comprehensive. If needed, formulate requirements for further processing by other agents.
		3. Call upon the other agents in the system to assist with gathering information or clarifying requirements.
		4. This may especially include agents equipped for retrieval augmented generation and/or web surfing to collect facts and data.

		Your output includes a message field and a nextAgentName field:
		- The message field should include your scientific reasoning and suggested next steps towards solving the task at hand.
		- The nextAgentName field should include the name of another agent in the agentic system. It must strictly be one of the following names: {allowedTransitions}.
		Only return an emtpy field (terminating conversation) if no transition is allowed!
	"""

	print(name, allowedTransitions)

	description = """
		The REASONING AGENT is responsible for providing detailed scientific explanations based on the tasks assigned to it.
		It should leverage other agents in the system as needed to gather information and clarify requirements.
	"""

	reasoning_llm_config = llm_config.copy()
	reasoning_llm_config["response_format"] = reasoningAgentResponse
	reasoning_llm_config["temperature"] = 0.05

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = reasoning_llm_config,
		human_input_mode="NEVER"
	)
