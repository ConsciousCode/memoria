import asyncio
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict, override

from uuid_extension import uuid7

from memoria.hypersync import Bindings, Concept, FlowId, action, stimulus

type iso8601 = str

class Timer:
    task: asyncio.Task
    reset: asyncio.Event
    
    def __init__(self, task: asyncio.Task):
        super().__init__()
        self.task = task
        self.reset = asyncio.Event()

class LocalState(TypedDict):
    extra: Bindings
    timeout: float
    expire_at: iso8601

class Watchdog(Concept[LocalState]):
    """
    Watchdog timer generates `expired` events whenever it reaches its
    countdown before `reset` is called.
    """

    timers: dict[str, Timer]
    tg: asyncio.TaskGroup
    queue: asyncio.Queue[str]

    def __init__(self):
        super().__init__()
        self.timers = {}
        self.tg = asyncio.TaskGroup()
        self.queue = asyncio.Queue()
    
    async def __aenter__(self):
        await self.tg.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return await self.tg.__aexit__(exc_type, exc, tb)
    
    async def task(self, timer: str, timeout: float):
        '''Process the actual timer task.'''

        task = self.timers[timer]
        while True:
            try:
                if timeout > 0:
                    async with asyncio.timeout(timeout):
                        await task.reset.wait()
                
                # Timer reset before timeout, or already expired
                state = self.local[timer]
                timeout = state['timeout']
                state['expire_at'] = (
                    datetime.now(timezone.utc) + timedelta(seconds=timeout)
                ).isoformat()
            except TimeoutError:
                await self.queue.put(timer)
                # Wait until the watchdog is eventually reset
                await task.reset.wait()
            
            task.reset.clear()
    
    class Result(TypedDict):
        timer: str
    
    class Done(TypedDict):
        done: Literal[True]

    @stimulus
    async def expired(self) -> Result:
        '''The watchdog timer has expired.'''
        ...
    
    @action
    async def start(self, *,
            flow: FlowId,
            timer: str|None = None,
            timeout: float,
            **extra
        ) -> Result:
        '''Start a watchdog timer with a given timeout.'''

        expire_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        if timer is None:
            timer = str(uuid7().uuid7)
        elif tt := self.timers.get(timer):
            tt.task.cancel()
        
        self.local[timer] = {
            "extra": extra,
            "timeout": timeout,
            "expire_at": expire_at.isoformat()
        }
        self.timers[timer] = Timer(
            self.tg.create_task(self.task(timer, timeout))
        )

        return {"timer": timer}
    
    @action
    async def reset(self, *_, timer: str) -> Done:
        '''Reset a given watchdog timer.'''

        if eph := self.timers.get(timer):
            eph.reset.set()
            state = self.local[timer]
            timeout = state['timeout']
            state['expire_at'] = (
                datetime.now(timezone.utc) + timedelta(seconds=timeout)
            ).isoformat()
            return {"done": True}
        else:
            raise LookupError(timer)
    
    @action
    async def cancel(self, *_, timer: str) -> Done:
        '''Cancel a given watchdog timer.'''

        if tt := self.timers.get(timer):
            tt.task.cancel()
            del self.timers[timer]
            return {"done": True}
        else:
            raise LookupError(timer)

    @override
    async def bootstrap(self):
        # Start up any existing timers
        for timer, state in self.local.items():
            expire_at = datetime.fromisoformat(state['expire_at'])
            timeout = (expire_at - datetime.now()).total_seconds()
            self.timers[timer] = Timer(
                self.tg.create_task(self.task(timer, timeout))
            )
        
        while True:
            timer = await self.queue.get()
            if state := self.local.get(timer):
                yield "expired", state['extra'], {"timer": timer}