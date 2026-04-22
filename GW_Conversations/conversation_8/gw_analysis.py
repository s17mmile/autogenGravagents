# filename: gw_analysis.py
import os
import sys
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
from pycbc.dataset import get_dataset_info
from gwosc import get_file_urls
from gwosc.timeline import get_segments

# Check for required libraries
try:
    import gwpy
    import pycbc
    import gwosc
except ImportError as e:
    print(f"Error: {e}. Please ensure that gwpy, pycbc, and gwosc are installed.")
    sys.exit(1)

# Set the working directory and parent directory for saving plots
working_dir = os.getcwd()
parent_dir = os.path.dirname(working_dir)

# Get event name from command line argument or use default
if len(sys.argv) > 1:
    event_name = sys.argv[1]
else:
    event_name = 'GW150914'

# 1. Query dataset information
print(f"Querying dataset information for {event_name}...")
try:
    dataset_info = get_dataset_info(event_name)
    print(f"Dataset Info: {dataset_info}")
except Exception as e:
    print(f"Failed to get dataset info: {e}")

# 2. Retrieve data file URLs
print(f"Retrieving data file URLs for {event_name}...")
try:
    data_file_urls = get_file_urls(event_name)
    print(f"Data File URLs: {data_file_urls}")
except Exception as e:
    print(f"Failed to retrieve data file URLs: {e}")

# 3. Query timeline segments
print(f"Querying timeline segments for {event_name}...")
try:
    timeline_segments = get_segments(event_name)
    print(f"Timeline Segments: {timeline_segments}")
except Exception as e:
    print(f"Failed to query timeline segments: {e}")

# 4. Use the low-level API to access specific data
print(f"Accessing data using low-level API for {event_name}...")
try:
    # Load strain data for a specific time segment
    strain_data = TimeSeries.fetch_open_data('H1', 1126259462, 1126259472)
    # Plotting the strain data
    print("Plotting strain data...")
    plot = Plot(strain_data)
    plot.title = f'Strain Data for {event_name}'
    plot.xlabel = 'Time (GPS)'
    plot.ylabel = 'Strain'
    plot.save(os.path.join(parent_dir, f'strain_data_{event_name}.png'))
    print(f"Strain data plot saved as 'strain_data_{event_name}.png' in the parent directory.")
except Exception as e:
    print(f"Failed to access or plot data: {e}")
