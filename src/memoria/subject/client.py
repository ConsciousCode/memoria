from typing import TYPE_CHECKING

from cachetools import LRUCache

if TYPE_CHECKING:
    import httpx
    from cid import CID

class SubjectClient:
    '''Client for the subject server.'''
    def __init__(self, client: 'httpx.AsyncClient'):
        self.client: 'httpx.AsyncClient' = client
        self.cache: LRUCache['CID', bytes] = LRUCache(100*1024*1024, len)

    async def ipfs(self, cid: 'CID'):
        """Fetch the content of the given CID from IPFS."""
        if cid in self.cache:
            return self.cache[cid]

        res = await self.client.get(f"/ipfs/{cid}")
        if res.is_error:
            return None

        data = res.content
        self.cache[cid] = data
        return data
