import asyncio
from datetime import datetime, timedelta, timezone
from typing import NotRequired, TypedDict, cast, override
from uuid import UUID

from uuid_extension import uuid7
from memoria.hypersync import Bindings, Concept, action, event, mutable_t

type iso8601 = str

class Timer(Concept):
    """Source of time-based events"""

    tasks: dict[UUID, asyncio.Task]
    tg: asyncio.TaskGroup
    queue: asyncio.Queue[UUID]

    def __init__(self):
        super().__init__()
        self.tg = asyncio.TaskGroup()
        self.queue = asyncio.Queue()
    
    async def __aenter__(self):
        await self.tg.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return await self.tg.__aexit__(exc_type, exc, tb)
    
    async def task(self, timer: UUID, next: datetime, interval: int|float):
        '''Process the actual timer task.'''
        now = datetime.now()
        td = (next - now).total_seconds()
        if td < 0:
            if interval:
                # Start is in the past, try to project a new one
                next += timedelta(seconds=interval*(1 + (-td)//interval))
                self.state[timer]['next'] = next.timestamp()
        else:
            await asyncio.sleep(td)
        
        await self.queue.put(timer)

        while interval:
            await asyncio.sleep(interval)
            await self.queue.put(timer)
        
        del self.state[timer]
    
    def normalize(self, this: str|None):
        if this is None:
            timer = uuid7().uuid7
        else:
            timer = UUID(this)
            if (task := self.tasks.get(timer)) is not None:
                task.cancel()
        
        return timer
    
    class Now(TypedDict):
        datetime: iso8601

    @action
    async def now(self) -> Now:
        return {"datetime": datetime.now(timezone.utc).isoformat()}
    
    class Construct(TypedDict):
        this: str
    
    @event
    async def trigger(self, *, name: str|None = None, timer: str):
        '''Event for when a timer is triggered.'''

    @action
    async def every(self, *,
            timer: str | None,
            name: str | None = None,
            interval: int|float
        ) -> Construct:
        '''Schedule a timer which triggers every interval.'''
        this = self.normalize(timer)
        start = datetime.now(timezone.utc) + timedelta(seconds=interval)
        state = {
            "start": start.isoformat(),
            "interval": interval
        }
        if name is not None:
            state['name'] = name
        
        self.state[this] = state

        self.tg.create_task(self.task(this, start, interval))

        return {"this": str(this)}
    
    @action
    async def after(self, *,
            this: str | None,
            name: str,
            delay: int|float
        ) -> Construct:
        '''Schedule a timer which triggers after a given delay.'''
        timer = self.normalize(this)
        start = datetime.now(timezone.utc) + timedelta(seconds=delay)
        state: dict[str, mutable_t] = {
            "start": start.isoformat()
        }
        if name is not None:
            state['name'] = name
        self.state[timer] = state

        self.tg.create_task(self.task(timer, start, 0))

        return {"this": str(timer)}
    
    @action
    async def at(self, this: str | None, name: str, time: iso8601) -> Construct:
        '''Schedule a timer for a specific time.'''
        timer = self.normalize(this)
        start = datetime.fromisoformat(time)
        state: dict[str, mutable_t] = {
            "start": time
        }
        if name is not None:
            state['name'] = name
        self.state[timer] = state

        self.tg.create_task(self.task(timer, start, 0))

        return {"this": str(timer)}
    
    @action
    async def cancel(self, this: str):
        '''Cancel a timer.'''
        timer = self.normalize(this)
        del self.state[timer]

        return {"success": True}

    @override
    async def bootstrap(self):
        while True:
            timer = await self.queue.get()
            if (state := self.state.get(timer)) is None:
                continue
            
            result: dict[str, mutable_t] = {"timer": str(timer)}
            if (name := state.get('name')) is not None:
                result['name'] = name
            
            yield "trigger", {}, result