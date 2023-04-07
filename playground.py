# set OpenAI api key and SerpAPI api key in environment variables
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "abf71d83dfe294730c303a14b9df11e24c4f39b4587742af4399385d366930d9"


# First, Close the open, AI language model to control the agent
llm = OpenAI(temperature=0)

# Second, load the tools you want to use
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, initialize the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now, you can use the agent to generate text
prompt = """
  what was the high temperature in SF yesterday in Fahrenheit?
  What is that number raised to the .023 power?
  """
agent.run(prompt)
