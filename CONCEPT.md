## Personalization

I want to create an agent that learns about me.

Every interaction should be passed to a background agent that will extract
knowledge that we learned about the user preferences and store in a memory.

When processing an input, we retrieve and preferences we have learned about
the user.

We need to understand the task or topic for an input. decide how to handle
the request. Then when constructing request to other tools, we utilize any
personalization information.

If we learn the wrong thing, or if the user's preferences change we need to
update our store of facts.

We build a profile on the user. For each input, we ask - what have we learned
about the user's preferences?

## Topics

Each time you talk to the assistant it is about a topic. You need to be explicit at first about changing topics.