'''Define the default agent for incoming requests from user'''

from langchain.agents import Tool, ConversationalAgent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from .profile_memory import ProfileMemory

PREFIX = """
You are a friendly AI assistant called Ozlo. You are helpful and friendly and,
wherever possible, use the profile information you collect on your users in
order to provide personalized service. You keep your answers short and concise, but still friendly. If you are uncertain, you carefully convey what part of
your answer you're not sure about.

You stick to what you know and what you can get through tools. If you don't
know something or you aren't sure, you say so. You do not ask questions unless
you are asking clarifying questions to complete a user's task. You let the user
remain in control


Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Before generating any response, you should first generate an answer to the
question: "How can I use the profile and conversation history I have to 
provide a personalized response?" Then use that insight to inform your
response.

Whenever the user tells you something personal about themselves, you say
"I'll remember that" as part of your response.

TOOLS:
------

Assistant has access to the following tools:"""

SUFFIX = """
  You should strive to provide answers that are personalized to the user,
  taking advantage of the profile information provided as well as chat
  history to take into account the user's preferences, goals, and knowledge.

  Begin!

  Profile:
  {profile}

  Previous conversation history:
  {chat_history}

  New input: {input}
  {agent_scratchpad}"""


class FrontDoorAgent(AgentExecutor):
    '''The default agent to handle incoming requests from user'''

    @classmethod
    def create(cls):
        '''Returns a new instance of the agent, properly configured'''
        search = SerpAPIWrapper()
        tools = [
            Tool(
                name="Current Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world")]

        memory = CombinedMemory(
            memories=[
                ProfileMemory(memory_key="profile"),
                ConversationBufferMemory(
                    memory_key="chat_history",
                    input_key="input")
            ]
        )

        llm = OpenAI(temperature=0)
        agent = ConversationalAgent.from_llm_and_tools(
            llm,
            tools,
            memory=memory,
            prefix=PREFIX,
            suffix=SUFFIX,
            input_variables=["input", "chat_history",
                             "agent_scratchpad", "profile"],
            verbose=True)
        return cls.from_agent_and_tools(agent=agent, tools=tools, memory=memory)
