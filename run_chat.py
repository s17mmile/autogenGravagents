import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config_4o_mini

os.system("clear")

# Set parameters for conversation execution
maxRounds = 20

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    conversationPath=os.path.join(os.path.dirname(__file__), "GW_Conversations"),
	llm_config=commercial_llm_config_4o_mini,
	maxRounds=maxRounds,
	trackTokens=False
)

