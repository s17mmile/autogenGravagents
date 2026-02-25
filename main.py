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

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
    configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    llm_config=llm_config,
    maxRounds=maxRounds
)

# query = input("Please enter your query: ")
# query = "Write a Python script that calculates the first n fibonacci numbers and plot the result against n for n up to 20. Save the resulting image as 'fibonacci_plot.png' and print the 150th fibonacci number."
# query = "Write a simple data anlysis pipeline that takes in a spectrum measured by an HPGe detector (given in a text file, name of this file should be taken as input) and fits gaussian peaks to the prominent peaks in the spectrum. The code should output a plot of the spectrum with the fitted peaks overlaid and a text file listing the energies and intensities of the fitted peaks."
query = "Provide a short summary of the 'Advancing AI-Scientist Understanding' Paper given in the document corpus."

# Start the conversation
flexibleChat.startConversation(query=query)