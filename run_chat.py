import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import *

os.system("clear")

# Set parameters for conversation execution
maxRounds = 100

basePath = os.path.join(os.path.dirname(__file__), "GW_Conversations")

counter = 1
while os.path.exists(os.path.join(basePath, f"conversation_{counter}")):
	counter += 1
conversationPath = os.path.join(basePath, f"conversation_{counter}")

# Instantiate chat instance based on agent config file
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/GWConfig.txt",
    conversationPath=conversationPath,
	llm_config=commercial_llm_config_4o_mini,
	maxRounds=maxRounds,
	trackTokens=False
)

# Not-quite-complete query history for the general purpose and specialized GW analysis task.

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

query3 = f"""
	Use the web surfing agent to go to https://gwpy.github.io/docs/2.1.2/ and click through every entry (and sub-entry in dropdowns) in the left sidebar for a proper overview of GWpy documentation.
	Keep track of the URL of each documentation page.
	Then, ingest every single web page with the RAG Agent for use as future reference.
	Once that is done, pass control back to the Human Agent.
	"""

# Whoopsies, had a compatibility failure with RAG agent on WSL. Not gonna re-run the websurfer, so just pasting in the retrieved URLs :)
query4 = r"""
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

query5 = f"""
	RAG Agent, ingest all of the following:
	https://gwpy.readthedocs.io/en/stable/timeseries/
	https://gwpy.readthedocs.io/en/stable/timeseries/io/
	https://gwpy.readthedocs.io/en/stable/timeseries/get/
	https://gwpy.readthedocs.io/en/stable/timeseries/plot/

	https://gwpy.readthedocs.io/en/stable/spectrum/
	https://gwpy.readthedocs.io/en/stable/spectrum/io/
	https://gwpy.readthedocs.io/en/stable/spectrum/filtering/

	https://gwpy.readthedocs.io/en/stable/spectrogram/

	https://gwpy.readthedocs.io/en/stable/signal/

	https://gwpy.readthedocs.io/en/stable/reference/gwpy.signal.filter_design.bandpass/#gwpy.signal.filter_design.bandpass

	https://gwpy.readthedocs.io/en/stable/plot/
	https://gwpy.readthedocs.io/en/stable/plot/gps/
	https://gwpy.readthedocs.io/en/stable/plot/colorbars/
	https://gwpy.readthedocs.io/en/stable/plot/legend/
	https://gwpy.readthedocs.io/en/stable/plot/log/
    https://gwpy.readthedocs.io/en/stable/plot/colors/
    https://gwpy.readthedocs.io/en/stable/plot/filter/

    https://gwpy.readthedocs.io/en/stable/detector/channel/
    https://gwpy.readthedocs.io/en/stable/logging/

    Only perform ingestions, do not answer any further queries.
	"""

query6 = f"""
	Search the web to find the documentation of GWOSC, crawl the gwosc documentation website in search of the most relevant library functions/APIs and ingest the relevant documentation with the RAG agent for future reference.
	"""

query7 = """
	Using the GW Coding Agent's knowledge about pycbc and gwpy gained from previous queries (accessible through the RAG agent), write a script that performs the following tasks in sequence and plots each result along the way. Plots should be saved in the parent directory of the current working dir with appropriate names.
    
	Task 1: Data fetching
		- Determine the start and end time of the GW150914 event. This gives you a single GPS time, so add the offsets yourself.
        - If the appropriate files (gwosc_gw150914_h1.hdf5 and gwosc_gw150914_l1.hdf5) already exist (they should be located in the parent folder of the current working directory), load the TimeSeries objects from disk.
		- Else, Download the L1 and H1 strain data for GW150914 over a 12-second window centered on the merger (8s before, 4s after). Plot the strain vs time, save the plot and write the original time series data to disk as HDF5 in the parent directory.
	Task 2: Data filtering
		- Whiten the data using the built-in GWpy whiten function. Plot and save the results.
		- Apply a band-pass filter between 30 and 250 Hz to the whitened data.
	Task 3: Q-Transform
		- Create the q_transform spectroscopy plot for both detectors' filtered data. Make sure there is normalised energy bar in the plot.
"""

flexibleChat.startConversation(query7)