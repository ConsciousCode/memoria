'''
Single-turn chat interpreter
'''

from pydantic_ai.agent import Agent

from memoria.interpreter import Interpreter
from memoria.subject.client import SubjectClient

class ChatInterpreter(Interpreter):
    def __init__(self, model: str, client: 'SubjectClient'):
        agent = Agent(model)
        super().__init__(agent, client)

    def act(self, prompt: str):
