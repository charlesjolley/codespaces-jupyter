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

## Topics

For any given input, and looking at the recent context, we need to determine
what other context to retrieve to be passed to the ExecutorAgent to decide how
to respond.

We need this context in order to decide what personalization information to
retrieve.
