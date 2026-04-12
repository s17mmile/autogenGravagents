import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config

os.system("clear")

# Set parameters for conversation execution
maxRounds = 20

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
	llm_config=commercial_llm_config,
	maxRounds=maxRounds,
	trackTokens=True
)

# query = input("Please enter your query: ")
# query = "Write a Python script that calculates the first n fibonacci numbers and plot the result against n for n up to 20. Save the resulting image as 'fibonacci_plot.png' and print the 150th fibonacci number."
query = r"A sample of $255 \mathrm{mg}$ of neon occupies $3.00 \mathrm{dm}^3$ at $122 \mathrm{K}$. Write and execute code that uses the perfect gas law to calculate the pressure of the gas."
messages, tokenUsage = flexibleChat.startConversation(query=query)
print(tokenUsage)

# Change working dir for testing
flexibleChat.setConversationPath(os.path.abspath("conversations/testingConversation2"))

query = r"A sample of $255 \mathrm{mg}$ of neon occupies $3.00 \mathrm{dm}^3$ at $122 \mathrm{K}$. Write and execute code that uses the perfect gas law to calculate the pressure of the gas."
messages, tokenUsage = flexibleChat.startConversation(query=query)
print(tokenUsage)