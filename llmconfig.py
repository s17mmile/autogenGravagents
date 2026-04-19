import os
from dotenv import load_dotenv
from autogen import LLMConfig

load_dotenv()

# LOCAL API
local_llm_config_4o_mini = {
                    "api_type": "openai", 
                    "model": "gpt-4o-mini",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL")
                    }

local_llm_config_5_nano = {
                    "api_type": "openai", 
                    "model": "gpt-5-nano",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL")
                    }

local_llm_config_codestral = {
                    "model": "mistral/codestral-2405", 
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL")
                    }

local_llm_config_mistral_small = {
                    "model": "mistral/mistral-small-latest",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL")
                    }

# COMMERCIAL API
commercial_llm_config_4o_mini = {
                        "api_type": "openai", 
                        "model": "gpt-4o-mini",
                        "api_key":os.getenv("OPENAI_API_KEY"),
                        "temperature": 0.01
                        }

commercial_llm_config_5_nano = {
                        "api_type": "openai",
                        "model": "gpt-5-nano",
                        "api_key":os.getenv("OPENAI_API_KEY")
                        }

commercial_llm_config_5_4 = {
                        "api_type": "openai", 
                        "model": "gpt-5.4",
                        "api_key":os.getenv("OPENAI_API_KEY")
                        }