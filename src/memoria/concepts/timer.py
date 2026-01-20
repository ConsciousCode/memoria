import asyncio
from datetime import datetime, timedelta, timezone
from typing import TypedDict, override
from uuid import UUID

from uuid_extension import uuid7

from memoria.hypersync import Concept, action, event

type iso8601 = str

class LocalState(TypedDict):
    name: str|None
    trigger_at: iso8601
    interval: float|None

class Timer(Concept[LocalState]):
    '''
    Generate time-based events based on an initial offset and then optionally
    repeating at a given interval.
    '''

    tasks: dict[UUID, asyncio.Task]
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
    
    async def task(self, timer: UUID, trigger_at: datetime, interval: float|None):
        '''Process the actual timer task.'''
        now = datetime.now()
        td = (trigger_at - now).total_seconds()
        if td < 0:
            if interval:
                # Trigger is in the past, try to project a new one
                interval = td + interval*(1 + (-td)//interval)
        else:
            interval = td

        while interval is not None:
            self.local[timer]['trigger_at'] = (
                datetime.now(timezone.utc) + timedelta(seconds=interval)
            ).isoformat()
            await asyncio.sleep(interval)
            await self.queue.put(timer)

            interval = self.local[timer]['interval']
        
        del self.local[timer]
        del self.tasks[timer]
    
    class Result(TypedDict):
        timer: str
    
    class Success(TypedDict):
        success: bool

    @event
    async def trigger(self) -> Result:
        '''Event for when a timer is triggered.'''
        ...
    
    @action
    async def schedule(self, *,
            timer: str|None = None,
            name: str|None = None,
            trigger_at: iso8601|None = None,
            delay: float|None = None,
            interval: float|None = None
        ) -> Result:
        '''Schedule a timer which triggers every interval.'''
        if timer is None:
            this = uuid7().uuid7
        else:
            this = UUID(timer)
            if (task := self.tasks.get(this)) is not None:
                task.cancel()

        if trigger_at is None:
            tat = datetime.now(timezone.utc)
        else:
            tat = datetime.fromisoformat(trigger_at)
        
        if delay is not None:
            tat += timedelta(seconds=delay)
        
        self.local[this] = {
            "name": name,
            "trigger_at": tat.isoformat(),
            "interval": interval
        }

        self.tg.create_task(self.task(this, tat, interval))

        return {"timer": str(this)}
    
    @action
    async def cancel(self, *, timer: str) -> Success:
        '''Cancel a timer.'''

        this = UUID(timer)
        if (task := self.tasks.get(this)) is not None:
            task.cancel()
        del self.local[this]

        return {"success": True}

    @override
    async def bootstrap(self):
        # Retart all the existing tasks
        for timer, state in self.local.items():
            self.tasks[timer] = self.tg.create_task(self.task(
                timer,
                datetime.fromisoformat(state['trigger_at']),
                state['interval']
            ))
        
        while True:
            timer = await self.queue.get()
            if (state := self.local.get(timer)) is None:
                continue
            
            yield "trigger", {}, {"timer": str(timer)}