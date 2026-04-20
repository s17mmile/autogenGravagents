import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import *

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
	# configPath="flexibleAgents/agentConfigs/defaultConfigMinusSurfer.txt",
	configPath="flexibleAgents/agentConfigs/basicAgent.txt",
    conversationPath=basePath,
	llm_config=local_llm_config_gemma,
	maxRounds=maxRounds,
	trackTokens=False
)

query1 = """Use the DocAgent to ingest and learn about the following resources, relevant for gravitational wave data analysis. After ingestion, answer the question "What are the main challenges in gravitational wave data analysis and how can they be addressed?" based on the ingested documents.
                               - https://gwpy.github.io/docs/1.0.0/examples/timeseries/public.html
                               - https://github.com/bilby-dev/bilby/blob/main/examples/gw_examples/data_examples/GW150914.py
                               - https://colab.research.google.com/github/gw-odw/odw-2023/blob/main/Tutorials/Day_1/Tuto_1.2_Open_Data_access_with_GWpy.ipynb#scrollTo=tcXQnfN0vvWt
                               - https://en.wikipedia.org/wiki/First_observation_of_gravitational_waves
                               - https://ccrg.rit.edu/research/area/gravitational-wave-data-analysis
                               - https://gwosc.org/software/
                               
                               Once this is done, give a summary of the most important libraries and paradigms used in gravitational wave data analysis. If any critical information is missing, try searching the web for it - else just give the summary immediately.
                               """

query2 = f"""
	The strain data for the GW150914 event is available in the execution direectory as txt files "gwosc_gw150914_h1.txt" and "gwosc_gw150914_l1.txt". Each file contains a small header:
	# Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
	# This file has 16384 samples per second
	# starting GPS 1126259447 duration 32
	Analogous in the the other file.

	Write code to read in the strain vs time data of both detectors and whiten each signal. Save the results as "H1_strain_whitened.png" and "L1_strain_whitened.png".
"""       

query3 = f"""
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

query4 = f"""
	The strain data for the GW150914 event is available in the execution direectory as txt files "gwosc_gw150914_h1.txt" and "gwosc_gw150914_l1.txt". Each file contains a small header:
	# 	# Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
	# 	# This file has 16384 samples per second
	# 	# starting GPS 1126259447 duration 32
	# 	Analogous in the the other file.

	Write a script that uses the PyCBC library to analyze the GW150914 gravitational wave event. The script should perform the following tasks:
	- Fetch and plot the strain vs time data for the GW150914 event for both the H1 and L1 detectors.
	- Whiten and re-plot the strain data for both detectors.
	- Apply a band-pass filter between 30 and 250 Hz to the whitened data and plot the results.
	- Generate and plot the power spectral density (PSD) of the original and whitened data for both detectors.
	- Generate the q-transform spectrograms for both detectors and identify the time-frequency region where the signal is most prominent.
	- Summarize the key characteristics of the GW150914 event based on the analysis,

	Unfortunately, my PyCBC installation is currently broken, so do not execute the code yourself, but just give me the script to do this analysis that I can run once I have fixed my PyCBC installation. Also, if you find that any critical information is missing to complete this task, explicitly state what information is missing and do not attempt to fill in the gaps yourself.
	Your RAG Agent has knowledge of Pycbc and gwpy documentation. Make sure to query the RAG Agent for information on the syntax of pycbc and related libraries!
	""" 

query5 = "What can you tell me about the functions provided by PyCBC for Gravitational Wave Data pre-processing? What about the function signatures and syntax needed to use them?"

query6 = r"""Assume that all gases are perfect and that data refer to 298.15 K unless otherwise stated. Calculate the change in chemical potential of a perfect gas when its pressure is increased isothermally from $1.8 \mathrm{~atm}$ to $29.5 \mathrm{~atm}$ at $40^{\circ} \mathrm{C}$."""

flexibleChat.startConversation(query6)