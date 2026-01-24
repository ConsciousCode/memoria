import asyncio
from typing import TypedDict, cast, override
from uuid import UUID

from aiohttp import web
from uuid_extension import uuid7

from memoria.hypersync import Bindings, Concept, action, stimulus

class Request(TypedDict):
    server: str
    method: str
    url: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes

class Response(TypedDict):
    status: int
    headers: dict[str, str]
    content_type: str | None
    charset: str | None
    text: str | None

class LocalState(TypedDict):
    host: str
    port: int

class HTTP(Concept):
    """Chat completions from LLM providers."""

    tg: asyncio.TaskGroup
    queue: asyncio.Queue[tuple[str, Request]]
    reqs: dict[str, asyncio.Future[web.Response]]
    servers: dict[str, asyncio.Task]

    def __init__(self):
        super().__init__()
        self.tg = asyncio.TaskGroup()
        self.queue = asyncio.Queue()
        self.reqs = {}
        self.servers = {}
    
    async def __aenter__(self):
        await self.tg.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return await self.tg.__aexit__(exc_type, exc, tb)
    
    async def run_server(self, server: str, host: str, port: int):
        '''Run an HTTP server as a task to push requests to queue.'''
        async def handler(request: web.Request) -> web.Response:
            body = await request.read()

            uid = str(uuid7().uuid7)
            future = asyncio.Future()
            self.reqs[uid] = future
            q = request.rel_url.query
            await self.queue.put((uid, {
                "server": server,
                "method": request.method,
                "url": str(request.url),
                "path": request.path,
                "query": {k: q.getall(k) for k in q},
                "headers": dict(request.headers),
                "body": body
            }))
            try:
                async with asyncio.timeout(30):
                    return await future
            except TimeoutError:
                return web.Response(
                    status=504,
                    text="Response action not invoked."
                )

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

    class ReqResult(TypedDict):
        request: str

    @stimulus
    async def request(self, *,
            method: str,
            url: str,
            path: str,
            query: dict[str, list[str]],
            headers: dict[str, str],
            body: bytes
        ) -> ReqResult:
        '''An HTTP request to be processed by the server.'''
        ...
    
    @action
    async def respond(self, *,
            request: str,
            status: int = 200,
            headers: dict[str, str] | None = None,
            content_type: str | None = None,
            charset: str | None = None,
            text: str | None = None
        ):
        '''Respond to an HTTP request.'''
        if req := self.reqs.get(request):
            req.set_result(web.Response(
                status=status,
                headers=headers,
                content_type=content_type,
                charset=charset,
                text=text
            ))
            del self.reqs[request]
            return {"done": True}
        else:
            raise LookupError(request)
    
    class Create(TypedDict):
        server: str

    @action
    async def start(self, *,
            server: str|None = None,
            host: str = "0.0.0.0",
            port: int
        ) -> Create:
        '''Start a new HTTP server.'''
        if server is None:
            server = str(uuid7().uuid7)
        if task := self.servers.get(server):
            task.cancel()
        
        self.local[server] = {
            "host": host,
            "port": port
        }
        self.servers[server] = self.tg.create_task(
            self.run_server(server, host, port)
        )
        return {"server": server}
    
    @override
    async def bootstrap(self):
        for server, state in self.local.items():
            self.tg.create_task(
                self.run_server(server, state['host'], state['port'])
            )
        
        while True:
            req, body = await self.queue.get()
            print(req, body)
            yield "request", cast(Bindings, body), {"request": req}