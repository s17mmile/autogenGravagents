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
                                    "base_url":os.getenv("IZ_BASE_URL")})



# 3. Create our LLM agent
my_agent = ConversableAgent(
    name="helpful_agent",
    system_message="You are a poetic AI assistant, respond in rhyme.",
    llm_config=llm_config
)

# 4. Run the agent with a prompt
response = my_agent.run(
    message=initial_query,
    max_turns=3,
)

# 5. Iterate through the chat automatically with console output
response.process()

# 6. Print the chat
print(response.messages)