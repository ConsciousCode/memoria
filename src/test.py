import asyncio
from typing import TypedDict
from memoria.hypersync import Engine, Sync, Concept, Var, When, Then, action, stimulus

import aioconsole

class Stdio(Concept):
    name = "stdio"
    purpose = "Interface with stdio"

    class Input(TypedDict):
        data: str

    @stimulus
    async def input(self) -> Input:
        ...
    
    @action
    async def output(self, data: str):
        print(data)
        return {"success": True}

    async def bootstrap(self):
        while True:
            line: str = await aioconsole.ainput()
            yield "stdio/input", {}, {"data": line}

echo = Sync(
    "echo", "Echoes stdin to stdout",
    when=[
        When("stdio/input", {}, {"data": Var("data")})
    ],
    then=[
        Then("stdio/output", {"data": Var("data")})
    ]
)

async def main():
    engine = Engine(
        state={},
        concepts=[Stdio()],
        syncs=[echo]
    )
    await engine.run()

asyncio.run(main())