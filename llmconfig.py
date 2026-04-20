import os
from dotenv import load_dotenv
from autogen import LLMConfig

load_dotenv()

# LOCAL API
local_llm_config_4_1_nano = {
                    "api_type": "openai", 
                    "model": "gpt-4.1-nano",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "temperature": 0.01
                    }

local_llm_config_4_1_mini = {
                    "api_type": "openai", 
                    "model": "gpt-4.1-mini",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "temperature": 0.01
                    }

local_llm_config_4o_mini = {
                    "api_type": "openai", 
                    "model": "gpt-4o-mini",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "temperature": 0.01
                    }

local_llm_config_5_nano = {
                    "api_type": "openai", 
                    "model": "gpt-5-nano",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "temperature": 0.01
                    }

# Gemma pricing not listed in the LiteLLM cost map, using estimate pricing for same model through bedrock.
local_llm_config_gemma = {
    "api_type": "openai", 
    "model": "openai/gemma-4-31B-it-Q4_K_M.gguf",
    "api_key": os.getenv("IZ_API_KEY"),
    "base_url": os.getenv("IZ_BASE_URL"),
    "temperature": 0.01,
    "price": [0.00014,0.00038]
}

# Note: mistral tends to throw errors somewhere deep in the experimental DocAgent stack because some message is tagged with the wrong role.
# Ok and now mistral doesn't generate any replies on the LiteLLM proxy. Yikes. Retracting all tests and using other GPTs.
local_llm_config_codestral = {
                    "api_type": "mistral",
                    "model": "mistral/codestral-2405", 
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "price": [0.001, 0.003]
                    }

local_llm_config_mistral_small = {
                    "api_type": "mistral",
                    "model": "mistral/mistral-small-latest",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "price": [0.00006, 0.00018] 
                    }

local_llm_config_qwen_instruct = {
                    "api_type": "",
                    "model": "mistral/mistral-small-latest",
                    "api_key":os.getenv("IZ_API_KEY"),
                    "base_url":os.getenv("IZ_BASE_URL"),
                    "price": [0.00006, 0.00018] 
                    }


# COMMERCIAL API
commercial_llm_config_4o_mini = {
                        "api_type": "openai", 
                        "model": "gpt-4o-mini",
                        "api_key":os.getenv("OPENAI_API_KEY"),
                        "temperature": 0.01
                        }

commercial_llm_config_4_1_nano = {
                    "api_type": "openai", 
                    "model": "gpt-4.1-nano",
                    "api_key":os.getenv("OPENAI_API_KEY"),
                    "temperature": 0.01
                    }

commercial_llm_config_mistral_small = {
                    "api_type": "mistral",
                    "model": "mistral/mistral-small-latest",
                    "api_key":os.getenv("MISTRAL_API_KEY")
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

# Price estimate as above
commercial_llm_config_gemma = {
    "api_type": "openai", 
    "model": "openai/gemma-4-31B-it-Q4_K_M.gguf",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.01,
    "price": [0.00014,0.00038]
}