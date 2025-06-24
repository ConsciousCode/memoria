'''
Common server utilities.
'''
from contextlib import contextmanager
from typing import Optional
from fastapi import Request
from fastmcp import Context

from db import Database
from memoria import Memoria

def mcp_context(ctx: Context|Request) -> Memoria:
    '''Get memoria from the FastAPI context.'''
    if isinstance(ctx, Context):
        return ctx.request_context.lifespan_context
    else:
        return ctx.app.state.lifespan_context

_gmemoria: Optional[tuple[Memoria, int]] = None

@contextmanager
def lifespan():
    '''Lifespan context for the FastAPI app.'''
    global _gmemoria
    if _gmemoria:
        m, count = _gmemoria
        _gmemoria = (m, count + 1)
        yield m
    else:
        with Database("private/memoria.db", "files") as db:
            m = Memoria(db)
            _gmemoria = (m, 1)
            yield m

    m, count = _gmemoria
    _gmemoria = (m, count - 1) if count > 1 else None