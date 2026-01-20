import asyncio
from datetime import datetime, timedelta, timezone
from typing import Type, TypedDict, override
from uuid import UUID

from uuid_extension import uuid7

from memoria.hypersync import Concept, action, event

type iso8601 = str

class Tasks:
    task: asyncio.Task
    reset: asyncio.Event
    
    def __init__(self, task: asyncio.Task):
        super().__init__()
        self.task = task
        self.reset = asyncio.Event()

class LocalState(TypedDict):
    name: str|None
    timeout: float
    expire_at: iso8601

class Watchdog(Concept[LocalState]):
    """Source of time-based events"""

    tasks: dict[UUID, Tasks]
    tg: asyncio.TaskGroup
    queue: asyncio.Queue[UUID]

    def __init__(self):
        super().__init__()
        self.tasks = {}
        self.tg = asyncio.TaskGroup()
        self.queue = asyncio.Queue()
    
    async def __aenter__(self):
        await self.tg.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return await self.tg.__aexit__(exc_type, exc, tb)
    
    async def task(self, timer: UUID, timeout: float):
        '''Process the actual timer task.'''

        task = self.tasks[timer]
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
    
    class Success(TypedDict):
        success: bool

    @event
    async def expired(self) -> Result:
        '''The watchdog timer has expired.'''
        ...
    
    @action
    async def start(self, *,
            name: str|None = None,
            timer: str|None = None,
            timeout: float
        ) -> Result:
        '''Start a watchdog timer with a given name and timeout.'''

        expire_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        if timer is None:
            this = uuid7().uuid7
        else:
            this = UUID(timer)
            if state := self.local.get(this):
                if name is None:
                    name = state['name']
        
        self.local[this] = {
            "name": name,
            "timeout": timeout,
            "expire_at": expire_at.isoformat()
        }
        self.tasks[this] = Tasks(
            self.tg.create_task(self.task(this, timeout))
        )

        return {"timer": str(this)}
    
    @action
    async def reset(self, timer: str) -> Success:
        '''Reset a given watchdog timer.'''

        this = UUID(timer)
        if eph := self.tasks.get(this):
            eph.reset.set()
            state = self.local[this]
            timeout = state['timeout']
            state['expire_at'] = (
                datetime.now(timezone.utc) + timedelta(seconds=timeout)
            ).isoformat()
            return {"success": True}
        else:
            raise LookupError(this)
    
    @action
    async def cancel(self, timer: str) -> Success:
        '''Cancel a given watchdog timer.'''

        this = UUID(timer)
        if tt := self.tasks.get(this):
            tt.task.cancel()
            del self.tasks[this]
            return {"success": True}
        else:
            raise LookupError(this)

    @override
    async def bootstrap(self):
        # Start up any existing timers
        for timer, state in self.local.items():
            expire_at = datetime.fromisoformat(state['expire_at'])
            timeout = (expire_at - datetime.now()).total_seconds()
            self.tasks[timer] = Tasks(
                self.tg.create_task(self.task(timer, timeout))
            )
        
        while True:
            timer = await self.queue.get()
            if state := self.local.get(timer):
                yield "expired", {}, {"timer": str(timer)}