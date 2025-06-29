'''
Common server utilities.
'''
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI
from fastmcp import FastMCP

from ipld.cid import CID
from ipld.ipfs import Blocksource, Blockstore, CompositeBlocksource, FlatfsBlockstore
from src.memoria import Memoria, Database
from src.ipld import dagcbor, dagjson, dagpb

class AppState:
    '''Application state for the FastAPI app.'''
    def __init__(self, blockstore: Blockstore, memoria: Memoria):
        self.store: Blockstore = blockstore
        self.memoria: Memoria = memoria
        self.blocksource: Blocksource = CompositeBlocksource(
            memoria, blockstore
        )
    
    def dag_get(self,
        cid: CID,
        output_codec: str = "dag-json"
    ) -> Optional[str|bytes]:
        '''Internal function to retrieve a block from the IPFS DAG.'''
        memoria = self.memoria
        bsource = self.blocksource

        if cid == 'bafkqaaa':
            return b''
        
        if ob := memoria.lookup_memory(cid):
            ob = ob.model_dump()
        elif ob := memoria.lookup_memory(cid):
            ob = ob.model_dump()
        elif ob := bsource.dag_get(cid):
            match cid.codec:
                case 'dag-pb':
                    ob = dagpb.unmarshal(ob)
                    ob = {
                        "Links": [
                            {
                                "Name": link.Name,
                                "Hash": str(CID(link.Hash)),
                                "Size": link.ByteSize()
                            } for link in ob.Links
                        ],
                        "Data": ob.Data
                    }
                case 'dag-cbor': ob = dagcbor.unmarshal(ob)
                case 'dag-json': ob = dagjson.unmarshal(ob)
                case _:
                    return ob
        else:
            return None
        
        match output_codec:
            case "dag-cbor": return dagcbor.marshal(ob)
            case "dag-json": return dagjson.marshal(ob)
            case _: raise NotImplementedError()

@asynccontextmanager
async def root_lifespan(app: FastAPI):
    '''Lifespan context for the FastAPI app.'''
    with Database("private/memoria.db", "files") as db:
        app.state.data = AppState(
            FlatfsBlockstore("private/blocks"),
            Memoria(db)
        )
        yield

app = FastAPI(lifespan=root_lifespan)

@asynccontextmanager
async def subapp_lifespan(subapp: FastAPI):
    '''Lifespan context for a subapp.'''
    if not hasattr(app.state, "state"):
        raise RuntimeError("App state not initialized.")
    subapp.state.data = app.state.data
    yield
    del subapp.state.data

depend_appstate = Depends(lambda request: request.app.state.data)

@asynccontextmanager
async def mcp_lifespan(server: FastMCP):
    '''Lifespan context for the FastAPI app.'''
    yield app.state.data