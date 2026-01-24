from memoria.hypersync import Concept, action

class Test(Concept):
    """Test concept that does nothing."""

    @action
    async def nop(self):
        return {"done": True}
    
    @action
    async def echo(self, *, value: str):
        return {"value": value}