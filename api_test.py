from autogen import ConversableAgent, LLMConfig

from llmconfig import local_llm_config, commercial_llm_config

print("Building agent")

agent = ConversableAgent(
    name = "TestAgent",
    system_message = "You are a test agent for testing the API.",
    llm_config = local_llm_config,
    human_input_mode="always"
)

print("Running agent")

response = agent.run(message = "Hello, this is a test message.",
            max_turns=2)

response.process()