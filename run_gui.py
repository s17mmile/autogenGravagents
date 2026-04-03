from flexibleAgents.GUI import AgentChatGuiHandler
from llmconfig import local_llm_config, commercial_llm_config

# Set parameters for conversation execution
maxRounds = 5

print("Flexible chat GUI instance creating...")

# Instantiate chat instance based on agent config file WITH GUI ENABLED
flexibleChatGUI = AgentChatGuiHandler(
    configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    llm_config=commercial_llm_config,
    maxRounds=maxRounds
)

print("Flexible chat GUI instance created.")