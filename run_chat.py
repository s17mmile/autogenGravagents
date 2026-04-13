import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import local_llm_config, commercial_llm_config_4o_mini

os.system("clear")

# Set parameters for conversation execution
maxRounds = 50

basePath = os.path.join(os.path.dirname(__file__), "GW_Conversations")

counter = 1
while True:
	dirname = f"conversation_{counter}"
	if not os.path.exists(os.path.join(basePath, dirname)):
		break
	counter += 1
conversationPath = os.path.join(basePath, dirname)

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/defaultConfigMinusSurfer.txt",
    conversationPath=basePath,
	llm_config=commercial_llm_config_4o_mini,
	maxRounds=maxRounds,
	trackTokens=False
)

# query="""Use the DocAgent to ingest and learn about the following resources, relevant for gravitational wave data analysis. After ingestion, answer the question "What are the main challenges in gravitational wave data analysis and how can they be addressed?" based on the ingested documents.
#                                - https://gwpy.github.io/docs/1.0.0/examples/timeseries/public.html
#                                - https://github.com/bilby-dev/bilby/blob/main/examples/gw_examples/data_examples/GW150914.py
#                                - https://colab.research.google.com/github/gw-odw/odw-2023/blob/main/Tutorials/Day_1/Tuto_1.2_Open_Data_access_with_GWpy.ipynb#scrollTo=tcXQnfN0vvWt
#                                - https://en.wikipedia.org/wiki/First_observation_of_gravitational_waves
#                                - https://ccrg.rit.edu/research/area/gravitational-wave-data-analysis
#                                - https://gwosc.org/software/
                               
#                                Once this is done, give a summary of the most important libraries and paradigms used in gravitational wave data analysis. If any critical information is missing, try searching the web for it - else just give the summary immediately.
#                                """

# query = f"""
# 	The strain data for the GW150914 event is available in the execution direectory as txt files "gwosc_gw150914_h1.txt" and "gwosc_gw150914_l1.txt". Each file contains a small header:
# 	# Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
# 	# This file has 16384 samples per second
# 	# starting GPS 1126259447 duration 32
# 	Analogous in the the other file.

# 	Write code to read in the strain vs time data of both detectors and whiten each signal. Save the results as "H1_strain_whitened.png" and "L1_strain_whitened.png".
# """       

query = f"""
Ingest:
	https://pycbc.org/pycbc/latest/html/catalog.html#accessing-data-around-each-event
	https://pycbc.org/pycbc/latest/html/dataquality.html
	https://pycbc.org/pycbc/latest/html/frame.html
	https://pycbc.org/pycbc/latest/html/fft.html
	https://pycbc.org/pycbc/latest/html/gw150914.html
	https://pycbc.org/pycbc/latest/html/detector.html
	https://pycbc.org/pycbc/latest/html/psd.html
	https://pycbc.org/pycbc/latest/html/noise.html
	https://pycbc.org/pycbc/latest/html/waveform.html
	https://pycbc.org/pycbc/latest/html/filter.html
	https://pycbc.org/pycbc/latest/html/distributions.html
"""

flexibleChat.startConversation(query)