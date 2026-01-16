import asyncio
from memoria.hypersync import Engine, Sync, Concept, Var, When, Then, Module, action

import aioconsole

class Stdio(Concept):
    name = "stdio"
    purpose = "Interface with stdio"

    @action
    def input(self):
        return {"error": "Bootstrap only"}
    
    @action
    def output(self, data: str):
        print(data)
        return {"success": True}

echo = Sync(
    "echo", "Echoes stdin to stdout",
    when=[
        When("EchoTest.stdio/input", {}, {"line": Var("data")})
    ],
    then=[
        Then("EchoTest.stdio/output", {"data": Var("data")})
    ]
)

class EchoTest(Module):
    def __init__(self):
        super().__init__(
            "EchoTest", [Stdio], [echo]
        )

    async def bootstrap(self):
        while True:
            line: str = await aioconsole.ainput()
            yield "stdio/input", {}, {"line": line}

async def main():
    engine = Engine()
    async with engine.run():
        engine.load(EchoTest())

asyncio.run(main())