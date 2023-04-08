'''Memory for collecting personal information about the user'''

from datetime import datetime
from langchain.schema import BaseMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

OBSERVER_VARIABLES = ["input", "chat_history", "current_date"]
OBSERVER_TEMPLATE = """
    Below is a chat log between a human user and an AI. The Input is a new 
    message sent by the user to the chat. Profile context includes relevant observations we know about the user.

    Based on the new input, generate observations about who the user is as a person: their preferences, beliefs, personal details, etc. Focus on what you 
    can learn generally about the user and their personal traits. For any observations that are time dependent (such as a date, meeting, event, holiday, or milestone) start with "As of {current_date}"

    Output observations should be full sentences and refer to the human as "User",
    separated by newlines. Include as many relevant details as possible in each statement, including relevant context. Be sure to include any factual statements you can infer about the user. Be sure to include any preferences you can infer based on what and how they choose to do, buy, or travel to. Also include inferrences you can make that are likely about the user's personality based on observable facts.

    --EXAMPLES
    EXAMPLE 1:

    Input: Cappuccinos are my favorite coffee drink. I have them every day.

    Output:
    User's favorite coffee drink is a cappuccino
    User is a frequent coffee drinker

    EXAMPLE 2:

    Current Date: {current_date}

    Profile Context:
    User has a girlfriend named Tiffany

    Chat Log:

    Input: Tiffany's car has a flat! Help!

    Output:
    Tiffany (User's girlfriend) has a car
    As of {current_date}, Tiffany's (User's girlfriend) car has a flat tire
    User is asking for help fixing a flat.
    User's girlfriend might need immediate attention.

    Tiffany (User's girlfriend) has a car
    As of {current_date}, Tiffany's (User's girlfriend) car has a flat tire


    EXAMPLE 3:
    Profile Context:
    User has a girlfriend named Tiffany

    Chat Log:
    user: Tiffany's car has a flat!
    agent: I can help with that. What kind of car is it?

    Input: Lamborghini

    Output:
    Tiffany (User's girlfriend) has a Lamborghini
    Tiffany (User's girlfriend) is probably into nice cars

    Tiffany (User's girlfriend) has a Lamborghini
    Tiffany (User's girlfriend) is probably into nice cars
    --END EXAMPLES


    Current Date: {current_date}

    Profile Context:


    Chat Log:
    {chat_history}

    Input:
    {input}

    Output: 
    """

FILTER_IRRELEVANT_VARIABLES = ["statements"]
FILTER_IRRELEVANT_TEMPLATE = """
    Below is a list of statements learned about a user during chat. Output
    only those statements that are likely to be true in the future as well as
    right now. Omit statements that are only related to what the user was 
    looking for during the chat or are tied to a specific event. Do not modify any statements.

    Statements:
    {statements}

    Output:
"""

FIND_OUTDATED_VARIABLES = ["new_obs", "old_obs"]
FIND_OUTDATED_TEMPLATE = """
    Below are a list of new statements and old statements about a user. New 
    statements came later than the old statements and might reflect a change in
    status. Output the list of old statements that are definitely outdated as
    a result of the new statements. Only output those with high confidence.

    Output should be a JSON array of the type:
    [...[statement, reason]]
    where `statement` is the invalid statement and `reason` explains why you
    think it is invalid.

    New Statements:
    {new_obs}

    Old Statements:
    {old_obs}

    Output:
    """


class ProfileMemory(BaseMemory):
    """Memory class for collecting information about the user"""

    class Config:
        '''make pydantic happy'''
        extra = "allow"

    memory_key = "profile"
    llm = OpenAI(temperature=0)
    _db = Chroma.from_texts(["dummy"], collection_name="observations",
                            embedding=OpenAIEmbeddings())
    _stored_count = 1

    _observer_chain_inst = None
    _filter_chain_inst = None

    @property
    def memory_variables(self):
        return [self.memory_key]

    def load_memory_variables(self, inputs):
        '''Load profile data associated with the inputs'''
        db = self._db
        message = inputs["input"]
        k = min(self._stored_count, 5)
        obs = db.similarity_search(message, k)
        profile = "\n".join([doc.page_content for doc in obs])
        return {self.memory_key: profile}

    def save_context(self, inputs, _outputs):
        '''Collects any observations about the user'''

        # First find all the observations
        obs_chain = self._observer_chain
        chat_history = inputs["chat_history"] or ""
        message = inputs["input"]
        current_date = datetime.today().strftime('%Y-%m-%d')
        output = obs_chain.run(
            input=message,
            chat_history=chat_history,
            current_date=current_date
        )

        # Second, filter the uneccesary ones
        filter_chain = self._filter_chain
        new_obs = filter_chain.run(statements=output)

        # Finally, store new observations
        new_obs = new_obs.split("\n")
        self._db.add_texts(new_obs)
        self._stored_count = self._stored_count + len(new_obs)

    def clear(self):
        '''Clear the stored memory'''
        print("clear profile memory")

    @property
    def _observer_chain(self):
        if self._observer_chain_inst is None:
            llm = self.llm
            prompt = PromptTemplate(
                template=OBSERVER_TEMPLATE,
                input_variables=OBSERVER_VARIABLES
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            self._observer_chain_inst = chain
        return self._observer_chain_inst

    @property
    def _filter_chain(self):
        if self._filter_chain_inst is None:
            llm = self.llm
            prompt = PromptTemplate(
                template=FILTER_IRRELEVANT_TEMPLATE,
                input_variables=FILTER_IRRELEVANT_VARIABLES
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            self._filter_chain_inst = chain
        return self._filter_chain_inst
