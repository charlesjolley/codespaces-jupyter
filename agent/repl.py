'''Kicks off an interactive instance'''

import langchain
from .front_door import FrontDoorAgent

BASIC_PROFILE = "Hi. My name is Charles. I live in Manhattan Beach. I love espresso, mexican, and fine dining. I'm into Italian if it is very authentic."


def main():
    '''Launches the agent repl'''
    # langchain.verbose = True
    agent = FrontDoorAgent.create()
    agent.run(input=BASIC_PROFILE)

    print('Ozlo> Hi! How can I help!')
    while True:
        line = input('User> ')
        output = agent.run(input=line)
        print(f"Ozlo> {output}")
