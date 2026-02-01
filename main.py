from autogen import LLMConfig

from dotenv import load_dotenv
import os
load_dotenv()

from flexibleAgents import agentChat


# BEFORE RUNNING, MAKE SURE TO SET UP YOUR .env FILE WITH AN APPROPRIATE URL AND API KEY
# In my case, I am using ChatGPT-4o mini through the IZ VPN at the University of Bonn.

# Define LLM configuration to be used for all agent instantiations
# Different agents will add the output format and temperature to this llm config as needed.
llm_config = LLMConfig(config_list={"api_type": os.getenv("IZ_API_TYPE"), 
                                    "model": os.getenv("IZ_MODEL"),
                                    "api_key":os.getenv("IZ_API_KEY"),
                                    "base_url":os.getenv("IZ_BASE_URL")})

# Set parameters for conversation execution
maxRounds = 50

# Import chat based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
    configPath="flexibleAgents/agentConfigs/config1.txt",
    llm_config=llm_config,
    maxRounds=maxRounds
)

# query = input("Please enter your query: ")
query = "Write a Python script that calculates the first n fibonacci numbers and plot the result against n for n up to 10. Save the resulting image as 'fibonacci_plot.png' and print the resulting list of the first 10 fibonacci numbers."

# Start the conversation
flexibleChat.startConversation(query=query)