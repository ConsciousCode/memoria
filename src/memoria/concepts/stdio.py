from typing import Literal, TypedDict

import aioconsole

from memoria.hypersync import Concept, action, stimulus

# This might actually be inappropriate. Typically stdio interaction is opt-in,
# calling eg `input()`, but here input is exposed as events. It doesn't
# generally allow multiple consumers which means it also can't be actions. This
# all probably just means that stdio can't be its own concept and it has to be
# refactored into an explicit interface rather than a generic affordance.

class Stdio(Concept):
    """Interface with stdio"""

    class Input(TypedDict):
        text: str

    @action
    async def input(self, *, prompt: str="") -> Input:
        return {"text": await aioconsole.ainput(prompt)}
    
    class Done(TypedDict):
        done: Literal[True]
    
    @action
    async def print(self, *, data: str) -> Done:
        print(data)
        return {"done": True}

    async def bootstrap(self):
        while True:
            line: str = await aioconsole.ainput()
            yield "input", {}, {"data": line}