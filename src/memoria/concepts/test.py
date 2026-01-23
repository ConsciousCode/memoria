from memoria.hypersync import Concept, action

class Test(Concept):
    """Test concept that does nothing."""

    @action
    async def nop(self, **_):
        return {"done": True}
    
    @action
    async def echo(self, *, value: str, **_):
        return {"value": value}