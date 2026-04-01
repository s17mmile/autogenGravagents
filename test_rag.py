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

query = """
        Find information on the use of pycbc or similar tools for gravitational wave data analysis in academic papers or relevant documentation. Return the key takeaways for the usage of libraries for gravitational wave data analysis and the URLs of relevant papers or documentation. What alternative packages are there, and what do they do?
    """

# Start the conversation
flexibleChat.startConversation(query=query)