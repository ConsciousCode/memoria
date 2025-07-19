import httpx

from ipld.cid import CID

class InterpreterClient:
    '''
    Client for the interpreter server.
    '''
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def interpret(self, context: list[CID]):
        """
        Interpret the provided context and produce a new memory.
        """
        res = await self.client.post(
            "/interpret",
            json={"context": [str(cid) for cid in context]}
        )
        if res.is_error:
            raise Exception(f"Interpretation failed: {res.text}")
        return res.json()