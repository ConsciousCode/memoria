from typing import TypedDict
from memoria.hypersync import Concept, action, event

import aioconsole

class Stdio(Concept):
    """Interface with stdio"""

    class Input(TypedDict):
        data: str

    @event
    async def input(self) -> Input:
        ...
    
    @action
    async def output(self, data: str):
        print(data)
        return {"success": True}

    async def bootstrap(self):
        while True:
            line: str = await aioconsole.ainput()
            yield "input", {}, {"data": line}