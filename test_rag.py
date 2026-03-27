import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config

os.system("clear")

# Set parameters for conversation execution
maxRounds = 5

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
    configPath="flexibleAgents/agentConfigs/rag.txt",
    llm_config=commercial_llm_config,
    maxRounds=maxRounds
)

# query = input("Please enter your query: ")
query = """
    Provide a short summary of the 'Can theoretical physics research benefit from language agents' Paper given in the document corpus.
    """

# Start the conversation
flexibleChat.startConversation(query=query)