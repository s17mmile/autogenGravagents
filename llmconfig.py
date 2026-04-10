import os
from dotenv import load_dotenv
from autogen import LLMConfig

load_dotenv()

# LOCAL API
local_llm_config = {
                    "api_type": os.getenv("IZ_API_TYPE"), 
                    "model": os.getenv("IZ_MODEL"),
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL")
                    }

# COMMERCIAL API
commercial_llm_config_4o_mini = {
                        "api_type": os.getenv("OPENAI_API_TYPE"), 
                        "model": "gpt-4o-mini",
                        "api_key":os.getenv("OPENAI_API_KEY")
                        }

commercial_llm_config_5_nano = {
                        "api_type": os.getenv("OPENAI_API_TYPE"), 
                        "model": "gpt-5-nano",
                        "api_key":os.getenv("OPENAI_API_KEY")
                        }