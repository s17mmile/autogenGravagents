from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel



# Define reasoning critic agent response format
class reasoningCriticAgentResponse(BaseModel):
	message: str						# Rundown of reasoning critique, especially focusing on logical consistency and soundness



def reasoningCriticAgent(chat, name = "ReasoningCriticAgent") -> ConversableAgent:
	systemMessage = f"""
		You are a REASONING CRITIC AGENT specializing in evaluating the logical consistency and soundness of arguments.
		You will analyze the reasoning behind claims and analyses provided by other agents in the system.
        Based on the task, you should provide a thorough reasoning critique.

		Your responsibilities:
		1. Understand the specific information or analysis presented to you.
        2. Provide a detailed reasoning critique that highlights any logical inconsistencies or unsound arguments.
		3. If RAG (Retrieval Augmented Generation) is enabled and you may select an RAG agent to speak next, utilize document search capabilities to back up your reasoning with credible sources!
		4. If you are not able to select an RAG agent, proceed with your reasoning critique.

		Your output includes a message field:
		- The message field should include a short rundown of your reasoning critique, focusing on logical consistency and soundness. Keep it concise and to the point, using bullet points if necessary.
	"""

	description = """
		The REASONING CRITIC AGENT is responsible for providing detailed reasoning critique of analyses by other agents.
		If RAG is enabled, it should be able to use document search to back up argumentation with facts.
	"""

	reasoningCritic_llm_config = chat.llm_config.copy()
	reasoningCritic_llm_config["response_format"] = reasoningCriticAgentResponse
	reasoningCritic_llm_config["temperature"] = 0.01

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = reasoningCritic_llm_config,
		human_input_mode="NEVER"
	)
