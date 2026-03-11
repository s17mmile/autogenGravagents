from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv
import os

load_dotenv()

llm_config = LLMConfig({    "api_type": os.getenv("IZ_API_TYPE"), 
                            "model": os.getenv("IZ_MODEL"),
                            "api_key":os.getenv("IZ_API_KEY"),
                            "base_url":os.getenv("IZ_BASE_URL")})

print("Building agent")

agent = ConversableAgent(
    name = "TestAgent",
    system_message = "You are a test agent for testing the API.",
    llm_config = llm_config,
    human_input_mode="always"
)

print("Running agent")

response = agent.run(message = "Hello, this is a test message.",
            max_turns=2)

response.process()