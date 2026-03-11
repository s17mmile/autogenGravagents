from autogen import LLMConfig

from dotenv import load_dotenv
import os, warnings
load_dotenv()

from flexibleAgents import agentChat

# BEFORE RUNNING, MAKE SURE TO SET UP YOUR .env FILE WITH AN APPROPRIATE URL AND API KEY
# In my case, I am using ChatGPT-4o mini through the IZ VPN at the University of Bonn.

# Define LLM configuration to be used for all agent instantiations
# Different agents will add the output format and temperature to this llm config as needed.
llm_config = LLMConfig({"api_type": os.getenv("IZ_API_TYPE"), 
                            "model": os.getenv("IZ_MODEL"),
                            "api_key":os.getenv("IZ_API_KEY"),
                            "base_url":os.getenv("IZ_BASE_URL")})

# Set parameters for conversation execution
maxRounds = 50

# Instantiate chat instance based on agent config file WITH GUI ENABLED
flexibleChat = agentChat.flexibleAgentChat(
    configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    llm_config=llm_config,
    maxRounds=maxRounds,
    makeGUI=True
)