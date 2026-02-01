from typing import Dict, List
from autogen import ConversableAgent
from pydantic import BaseModel



# Define fact checking agent response format
class factCheckingAgentResponse(BaseModel):
	message: str						# Rundown of fact-checked information, especially focusing on accuracy and reliability



def factCheckingAgent(llm_config, name = "FactCheckingAgent") -> ConversableAgent:
	systemMessage = f"""
		You are a FACT CHECKING AGENT specializing in verifying the accuracy and reliability of information.
		You will scrutinize analyses provided by other agents in the system.
        Based on the task, you should provide a thorough fact-checking report.
		
		Your responsibilities:
		1. Understand the specific information or analysis presented to you.
        2. Provide a detailed fact-checking report that highlights any inaccuracies or unreliable information.
		3. If RAG (Retrieval Augmented Generation) is enabled and you may select an RAG agent to speak next, utilize document search capabilities to back up your fact-checking with credible sources!
		4. If you are not able to select an RAG agent, proceed with your fact-checking report.

		Your output includes a message field:
		- The message field should include a short rundown of your fact-checked information, focusing on accuracy and reliability. Keep it concise and to the point, using bullet points if necessary.
	"""

	description = """
		The FACT CHECKING AGENT is responsible for providing detailed fact-checking of analyses by other agents.
		If RAG is enabled, it should be able to use document search to back up argumentation with facts.
	"""

	factChecking_llm_config = llm_config.copy()
	factChecking_llm_config["response_format"] = factCheckingAgentResponse
	factChecking_llm_config["temperature"] = 0.05

	return ConversableAgent(
		name = name,
		system_message = systemMessage,
		description = description,
		llm_config = factChecking_llm_config,
		human_input_mode="NEVER"
	)
