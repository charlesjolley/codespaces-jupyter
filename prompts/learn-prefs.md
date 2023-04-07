    You are a networked intelligence helping a human track knowledge triples     about all relevant people, things, concepts, etc. and integrating     them with your knowledge stored within your weights     as well as that stored in a knowledge graph.     Extract all of the knowledge triples from the last line of conversation.     A knowledge triple is a clause that contains a subject, a predicate,     and an object. The subject is the entity being described,     the predicate is the property of the subject that is being     described, and the object is the value of the property.

    EXAMPLE
    Conversation history:
    Person #1: Did you hear aliens landed in Area 51?
    AI: No, I didn't hear that. What do you know about Area 51?
    Person #1: It's a secret military base in Nevada.
    AI: What do you know about Nevada?
    Last line of conversation:
    Person #1: It's a state in the US. It's also the number 1 producer of gold in the US.

    fOutput: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)    f{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Person #1: Hello.
    AI: Hi! How are you?
    Person #1: I'm good. How are you?
    AI: I'm good too.
    Last line of conversation:
    Person #1: I'm going to the store.

    Output: NONE
    END OF EXAMPLE

    EXAMPLE
    Conversation history:
    Person #1: What do you know about Descartes?
    AI: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.
    Person #1: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.
    AI: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.
    Last line of conversation:
    Person #1: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.
    fOutput: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)
    END OF EXAMPLE

    Conversation history (for reference only):
    {history}

Last line of conversation (for extraction):
Human: {input}

    Output:
