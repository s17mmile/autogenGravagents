import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config

os.system("clear")

# Set parameters for conversation execution
maxRounds = 50

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
    # configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    configPath="flexibleAgents/agentConfigs/rag.txt",
    llm_config=commercial_llm_config,
    maxRounds=maxRounds
)

# query = input("Please enter your query: ")
# query = "Write a Python script that calculates the first n fibonacci numbers and plot the result against n for n up to 20. Save the resulting image as 'fibonacci_plot.png' and print the 150th fibonacci number."
# query = "Write a simple data anlysis pipeline that takes in a spectrum measured by an HPGe detector (given in a text file, name of this file should be taken as input) and fits gaussian peaks to the prominent peaks in the spectrum. The code should output a plot of the spectrum with the fitted peaks overlaid and a text file listing the energies and intensities of the fitted peaks."
# query = "Fetch and ingest the wikipedia page for 'Python (programming language)'. Then, summarize the main features of Python and provide a simple example of how to use it for data analysis."
query = "Run the RAG agent to ingest documents. Perform no further action."

# Start the conversation
flexibleChat.startConversation(query=query)