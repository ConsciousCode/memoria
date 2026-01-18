import asyncio
from memoria.hypersync import Engine, Sync, Concept, Var, When, Then, action, event

import aioconsole

class Stdio(Concept):
    name = "stdio"
    purpose = "Interface with stdio"

    @event
    async def input(self):
        pass
    
    @action
    async def output(self, data: str):
        print(data)
        return {"success": True}

    async def bootstrap(self, state):
        self.state = state
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