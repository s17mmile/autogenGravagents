import os, warnings
from autogen import LLMConfig

from flexibleAgents import agentChat
from llmconfig import *

os.system("clear")

# Set parameters for conversation execution
maxRounds = 100

# Instantiate GW config with a full reset after each run
flexibleChat = agentChat.flexibleAgentChat(
	configPath="flexibleAgents/agentConfigs/GWConfig.txt",
	llm_config=commercial_llm_config_4o_mini,
	maxRounds=maxRounds,
	trackTokens=True,
	resetAfterConversation=True
)

# Extended query for testing consistency across runs. As this is a long and tough query, full completion is likely not going to happen - at least not every time.
query = """
	Using the GW Coding Agent's knowledge about pycbc and gwpy (accessible through the RAG agent), write a script that performs the following tasks in sequence and plots each result along the way. All plots should be saved in the current working directory with appropriate names.
    
	Task 1: Data fetching
		- Determine the start and end time of the GW150914 event. Fetching the event time gives you a single GPS time, so add the offsets yourself.
        - If the appropriate files (gwosc_gw150914_h1.hdf5 and gwosc_gw150914_l1.hdf5) already exist (they should be located in the parent folder of the current working directory), load the GWpy TimeSeries objects from disk.
		- Else, Download the L1 and H1 strain data for GW150914 over a 12-second window centered on the merger (8s before, 4s after). Plot the strain vs time, save the plot and write the original time series data to disk as HDF5 in the parent directory.
	
	Task 2: Data filtering
		- Whiten each detector's signal using TimeSeries.whiten(). Plot and save the results.
		- Apply a band-pass filter between 30 and 250 Hz to the whitened data. Plot and save the results.
	
	Task 3: Q-Transform
		- Create the q_transform spectroscopy plot for both detectors' filtered data. Make sure there is normalised energy bar in the plot, and set final color limits (0,25) in the plot.

	Task 4: Data format conversion
		- Convert both detectors' filtered GWpy TimeSeries data into the appropriate PyCBC data type for matched filtering. Ensure that both detectors have consistent sample rates and lengths, resampling or trimming as necessary.
		- Both the strain data and soon-to-be-created waveform templates need to be valid PyCBC TimeSeries objects with identical delta_t!

	Task 5: PyCBC template creation
		- Generate time-domain PyCBC waveform templates/models for identical component masses of 10, 20, 30, and 40 solar masses with zero spins. Specify a valid approximant (e.g., "SEOBNRv4_opt" or "IMRPhenomD") to avoid NoneType errors. Keep only templates longer than 0.2 s, and pad or truncate them to match the data length.
		- Scale each template so it matches the maximum strain amplitude.
		- For each template, create a plot overlaying it on to the strain data of each detector with high contrast. Save each plot as well as the template array data to the current working directory.

	Task 6: Calculating the Power Spectral Density
		- Calculate the Power Spectral Density (PSD) of the filtered data for each detector using PyCBC.
		- Interpolate and inverse spectrum truncate the PSD with a low frequency cutoff of 30 Hz to use it as a proper filter.
		- Plot the PSDs on a log-log scale and save the plots.

	Task 7: Matched filtering
		- Perform PyCBC matched filtering between each template and the strain data for both detectors. Plot the SNR time series for each template-detector pair, and save the plots.
		- Identify the template that yields the highest SNR peak in each detector, and report it as best fit.
"""

# Run ten tests into different outdirs
basePath = os.path.join(os.path.dirname(__file__), "GW_Conversations/consistency_test")

for i in range(10):
	input(f"Press Enter to start conversation {i}...")
	flexibleChat.setConversationPath(os.path.join(basePath, f"conversation_{i}"))
	flexibleChat.startConversation(query)
	input(f"Conversation {i} finished, record results! Then press Enter to continue...")