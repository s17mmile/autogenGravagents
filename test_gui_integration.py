print("imports starting")
from autogen import LLMConfig

from dotenv import load_dotenv
import os, warnings
load_dotenv()

print("imports done")
print("importing GUI Handler")

from flexibleAgents.GUI import AgentChatGuiHandler

print("GUI Handler imported")

# Define LLM configuration to be used for all agent instantiations
# Different agents will add the output format and temperature to this llm config as needed.
llm_config = LLMConfig({"api_type": os.getenv("IZ_API_TYPE"), 
                            "model": os.getenv("IZ_MODEL"),
                            "api_key":os.getenv("IZ_API_KEY"),
                            "base_url":os.getenv("IZ_BASE_URL")})

# Set parameters for conversation execution
maxRounds = 5

print("Flexible chat GUI instance creating")

# Instantiate chat instance based on agent config file WITH GUI ENABLED
flexibleChatGUI = AgentChatGuiHandler(
    configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    llm_config=llm_config,
    maxRounds=maxRounds
)

print("Flexible chat GUI instance created")