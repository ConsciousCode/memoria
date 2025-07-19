import httpx

from ipld.cid import CID

class SubjectClient:
    '''
    Client for the subject server.
    '''
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
    
    async def ipfs(self, cid: CID):
        """
        Fetch the content of the given CID from IPFS.
        """
        res = await self.client.get(f"/ipfs/{cid}")
        return None if res.is_error else res.content