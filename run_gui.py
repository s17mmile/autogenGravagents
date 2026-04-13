from flexibleAgents.GUI import AgentChatGuiHandler
from llmconfig import local_llm_config, commercial_llm_config_4o_mini

# Set parameters for conversation execution
maxRounds = 50

print("Flexible chat GUI instance creating...")

# Instantiate chat instance based on agent config file WITH GUI ENABLED
flexibleChatGUI = AgentChatGuiHandler(
    configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    llm_config=commercial_llm_config_4o_mini,
    maxRounds=maxRounds
)

print("Flexible chat GUI instance created.")