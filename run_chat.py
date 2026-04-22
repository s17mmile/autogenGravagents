import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import *

os.system("clear")

# Set parameters for conversation execution
maxRounds = 100

basePath = os.path.join(os.path.dirname(__file__), "GW_Conversations")

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/defaultConfig.txt",
    conversationPath=basePath,
	llm_config=commercial_llm_config_4o_mini,
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

query6 = f"""
	Use the web surfing agent to go to https://gwpy.github.io/docs/2.1.2/ and click through every entry in the left sidebar for a proper overview of GWpy documentation.
	Keep track of the URL of each documentation page. Include any sub-pages if they come up!
	Then, ingest each web page with the RAG Agent for use as future reference.
	Once that is done, pass control back to the Human Agent.
	"""

# Whoopsies, had a compatibility failure with RAG agent on WSL. Not gonna re-run the websurfer, so just pasting in the retrieved URLs :)
query7 = r"""
	RAG Agent! Ingest the webpages of all following URLs (found by webusrfer in previous run):

	Websurfer:
	{
    "extracted_content": [
        {
            "content": "\ud83d\udd17  Navigated to https://gwpy.github.io/docs/2.1.2/",
            "url": null
        },
        {
            "content": "\ud83d\udcc4  Extracted from page\n: ```json\n{\n  \"urls\": [\n    \"overview/\",\n    \"install/\",\n    \"citing/\",\n    \"timeseries/\",\n    \"spectrum/\",\n    \"spectrogram/\",\n    \"timeseries/statevector/\",\n    \"segments/\",\n    \"table/\",\n    \"signal/\",\n    \"plot/\",\n    \"cli/\",\n    \"detector/channel/\",\n    \"time/\",\n    \"astro/\",\n    \"env/\",\n    \"examples/timeseries/\",\n    \"examples/signal/\",\n    \"examples/frequencyseries/\",\n    \"examples/spectrogram/\",\n    \"examples/segments/\",\n    \"examples/table/\",\n    \"examples/miscellaneous/\",\n    \"dev/release/\",\n    \"genindex/\"\n  ]\n}\n```\n",
            "url": "https://gwpy.github.io/docs/2.1.2/"
        },
        {
            "content": "\ud83d\udd17  Opened new tab with https://gwpy.github.io/docs/2.1.2/overview/",
            "url": "https://gwpy.github.io/docs/2.1.2/"
        },
        {
            "content": "\ud83d\udd17  Opened new tab with https://gwpy.github.io/docs/2.1.2/install/",
            "url": "https://gwpy.github.io/docs/2.1.2/signal/"
        },
        {
            "content": "\ud83d\udd17  Opened new tab with https://gwpy.github.io/docs/2.1.2/citing/",
            "url": "https://gwpy.github.io/docs/2.1.2/signal/"
        },
        {
            "content": "\ud83d\udd17  Opened new tab with https://gwpy.github.io/docs/2.1.2/timeseries/",
            "url": "https://gwpy.github.io/docs/2.1.2/timeseries/"
        },
        {
            "content": "\ud83d\udd17  Opened new tab with https://gwpy.github.io/docs/2.1.2/spectrum/",
            "url": "https://gwpy.github.io/docs/2.1.2/table/"
        }
    ],}

	"""

query8 = """
	Using the knowledge about pycbc and gwpy gained from previous queries (accessible through the RAG agent), write a script that performs the following tasks in sequence and plots each result along the way. All plots should be saved to disk in the working directory with appropriate names. Any code written should include occasional print statements and GWpy logging to indicate task progress.

	If the coding agent ever gets stuck on pycbc or GWpy syntax, call the RAG Agent to retrieve documentation or code examples from ingested documents. Only call web search if this fails and the user explicitly authorizes it.
    
    Make sure to properly indent your code, especially in try/except blocks! This failed previously as the h1 data fetching line was improperly indented.
    
	Task 1: Data fetching
		- Determine the start and end time of the GW150914 event using gwosc event_gps().
        - If the appropriate file (GW150914_L1.txt) already exists, load it using TimeSeries.read(filename).
		- Else, Download the L1 and H1 strain data for GW150914 using TimeSeries.fetch_open_data() over a 12-second window centered on the merger (8s before, 4s after). Plot the strain vs time, save the plot and write the original data to disk using TimeSeries.write('data.txt').
	Task 2: Data filtering
		- Whiten the data using the built-in GWpy function. Plot and save the results.
		- Apply a band-pass filter between 30 and 250 Hz to the whitened data.
	Task 3: Q-Transform
		- Create the q_transform spectroscopy plot for both detectors' filtered data. make sure there is normalised energy bar in the plot.
	Task 4: Template creation
		- Use only the H1 data for this.
		- Generate PyCBC waveform templates for component masses 10-30 solar masses, zero spins, and specify a valid approximant (e.g., "SEOBNRv4_opt" or "IMRPhenomD") to avoid NoneType errors. Keep only templates longer than 0.2s, and pad or truncate them to match the data length. Convert both the strain data and templates to PyCBC TimeSeries with identical delta_t. Important: Before plotting, scale each template so that its maximum absolute amplitude matches the maximum absolute amplitude of the processed H1 strain. This ensures that the template is clearly visible when overlaid on the detector signal. For each template, create a separate plot overlaying the scaled template on top of the H1 strain signal, so the alignment is clearly visible. Additionally, produce a plot showing the combined H1 strain data for reference. Save all individual template overlay plots, the combined H1 strain plot, and the template arrays to disk for later analysis. Skip PSD estimation and matched filtering for now, but ensure all valid templates are included in the overlay plots.
"""

flexibleChat.startConversation(query8)