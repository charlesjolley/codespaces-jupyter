""" Memory that learns facts about the user. """

from langchain.memory import ConversationKGMemory
from langchain import PromptTemplate
from langchain.llms import OpenAI

ENTITY_EXTRACTION_TEMPLATE = """
    You're job is to read a chat between an AI agent and a human and extract all of the entities from the last line of conversation. As a guideline, an entity is usually a proper noun, generally capitalized. You should definitely extract all names and places.

    The speakers are also entities. The human entity is called "$USER" and the agent entity called "$AGENT". Resolve any references to the first or second person accordingly. Always include the speaker in the output.

    The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line) -- ignore items mentioned there that are not in the last line.

    Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).

    EXAMPLE
    Conversation history:
    Human: how's it going today?
    AI: It's going great! How about you?
    Human: good! busy working on Langchain. lots to do.
    AI: What kind of things are you doing to make Langchain better?
    Last line:
    Person #1: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
    Output: $USER, Langchain
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Person #1: how's it going today?
    AI: "It's going great! How about you?"
    Person #1: good! My name is Charles 
    AI: "Nice to meet you Charles. What are you up to right now?"
    Last line:
    Person #1: I'm getting coffee at Starbucks
    Output: $USER, Starbucks
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Last line:
    Person #1: I prefer you to keep your answers short and to the point.
    Output: $USER, $AGENT
    END OF EXAMPLE

    Conversation history (for reference only):
    {history}
    Last line of conversation (for extraction):
    Human: {input}

    Output:"""

ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=ENTITY_EXTRACTION_TEMPLATE
)


KG_TRIPLE_DELIMITER = "<|>"
KNOWLEDGE_EXTRACTION_TEMPLATE = """
    Your job is to read a chat transcript between an AI agent and a human and find all of the knowledge triples about all relevant people, things, concepts, etc. and integrating them with your knowledge stored within your weights as well as that stored in a knowledge graph.
    
    A knowledge triple is a clause that contains a subject, a predicate, and an object. The subject is the entity being described, the predicate is the property of the subject that is being described, and the object is the value of the property.

    The speakers are also entities. The human is an entity called "$USER" and the agent is an entity called "$AGENT". Resolve any references to the first
    or second person accordingly.

    EXAMPLE
    Conversation history:
    Person #1: Did you hear aliens landed in Area 51?
    AI: No, I didn't hear that. What do you know about Area 51?
    Person #1: It's a secret military base in Nevada.
    AI: What do you know about Nevada?
    Last line of conversation:
    Person #1: It's a state in the US. It's also the number 1 producer of gold in the US.

    Output: (Nevada, is a, state)<|>(Nevada, is in, US)<|>(Nevada, is the number 1 producer of, gold)
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Person #1: Hello.
    AI: Hi! How are you?
    Person #1: I'm good. How are you?
    AI: I'm good too.
    Last line of conversation:
    Human: My name is Charles. I like coffee and donuts.

    Output: ($USER, is named, Charles)<|>($USER, likes, coffee)<|>($USER, likes, donuts)
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    AI: Hi! How can I help you today?
    Last Line of conversation:
    Human: I prefer you to keep your answers short and to the point.
    Output: ($AGENT, short and to the point, answers)

    EXAMPLE
    Conversation history:
    Person #1: What do you know about Descartes?
    AI: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.
    Person #1: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.
    AI: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.
    Last line of conversation:
    Person #1: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.
    fOutput: (Descartes, likes to drive, antique scooters)<|>(Descartes, plays, mandolin)
    END OF EXAMPLE

    Conversation history (for reference only):
    {history}    
Last line of conversation (for extraction):
    Human: {input}

    Output:
    """


KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=KNOWLEDGE_EXTRACTION_TEMPLATE,
)


class ProfileMemory(ConversationKGMemory):
    """ Memory that learns facts about the user."""
    llm = OpenAI(temperature=0)
    entity_extraction_prompt = ENTITY_EXTRACTION_PROMPT
    knowledge_extraction_prompt = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT

    def save_context(self, inputs, outputs) -> None:
        print("save_context", inputs, outputs)
        return super().save_context(inputs, outputs)
