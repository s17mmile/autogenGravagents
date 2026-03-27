import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config

os.system("clear")

# Set parameters for conversation execution
maxRounds = 5

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
    configPath="flexibleAgents/agentConfigs/websurfer.txt",
    llm_config=commercial_llm_config,
    maxRounds=maxRounds
)

# query = input("Please enter your query: ")
query = """
    Find and download the paper \"Multi-Agent System for Cosmological Parameter Analysis\" as a PDF.
    """

# Start the conversation
flexibleChat.startConversation(query=query)