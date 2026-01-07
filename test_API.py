import openai

client = openai.OpenAI(
api_key="sk-3fvAX2H3EZbMG__xsOCtNA",

base_url="http://131.220.150.238:8080" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
# base_url="https://131.220.250.238:8080/"


)

response = client.chat.completions.create(
model="openai/gpt-4o-mini", # model to send to the proxy
messages = [
{
"role": "user",
"content": "this is a test request, write a short poem"
}
]
)

print(response)