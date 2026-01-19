from autogen import LLMConfig

from dotenv import load_dotenv
import os
load_dotenv()


# BEFORE RUNNING, MAKE SURE TO SET UP YOUR .env FILE WITH AN APPROPRIATE URL AND API KEY
# In my case, I am using ChatGPT-4o mini through the IZ VPN at the University of Bonn.

# Define LLM configuration to be used for all agent instantiations
llm_config = LLMConfig(config_list={"api_type": os.getenv("IZ_API_TYPE"), 
                                    "model": os.getenv("IZ_MODEL"),
                                    "api_key":os.getenv("IZ_API_KEY"),
                                    "base_url":os.getenv("IZ_BASE_URL")})



# Define consistent message/output structure for all agents



# Set parameters for conversation execution (maximum turns etc)



# Instantiate agents by reading agent config file