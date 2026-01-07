from autogen import ConversableAgent, LLMConfig

from dotenv import load_dotenv
import os
load_dotenv()





# Take initial query to be passed into the agentic system
initial_query = input("Enter the initial query for the agent: ")

# Or use a default query if none is provided
if not initial_query.strip():
    initial_query = "Write a short poem about the beauty of nature."



# Define our LLM configuration for OpenAI's GPT-4o mini running through the IZ VPN
llm_config = LLMConfig(config_list={"api_type": "openai", 
                                    "model": "gpt-4o-mini",
                                    "api_key":os.getenv("IZ_API_KEY"),
                                    "base_url":os.getenv("BASE_URL")})



# Create an Agent for each given System Message in the appropriate folder. This allows for easy scalability and expansion of the agent system.