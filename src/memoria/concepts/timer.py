import asyncio
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict, override
from uuid import UUID

from uuid_extension import uuid7

from memoria.hypersync import Bindings, Concept, FlowId, action, event

type iso8601 = str

class LocalState(TypedDict):
    extra: Bindings
    trigger_at: iso8601
    interval: float|None

class Timer(Concept[LocalState]):
    '''
    Generate timer events based on an initial offset and then optionally
    repeating at a given interval.
    '''

    timers: dict[str, asyncio.Task]
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
    
    async def task(self, timer: str, trigger_at: datetime, interval: float|None):
        '''Process the actual timer task.'''
        now = datetime.now()
        td = (trigger_at - now).total_seconds()
        if td < 0:
            if interval:
                # Trigger is in the past, try to project a new one
                interval = td + interval*(1 + (-td)//interval)
        else:
            interval = td

        # Repeat if there's an interval (or it's in the future)
        while interval is not None:
            self.local[timer]['trigger_at'] = (
                datetime.now(timezone.utc) + timedelta(seconds=interval)
            ).isoformat()
            await asyncio.sleep(interval)
            await self.queue.put(timer)

            interval = self.local[timer]['interval']
        
        del self.local[timer]
        del self.timers[timer]
    
    class Result(TypedDict):
        timer: str
    
    class Done(TypedDict):
        done: Literal[True]

    @event
    async def trigger(self, **_) -> Result:
        '''Event for when a timer is triggered.'''
        ...
    
    @action
    async def schedule(self, *,
            flow: FlowId,
            timer: str|None = None,
            trigger_at: iso8601|None = None,
            delay: float|None = None,
            interval: float|None = None,
            **extra
        ) -> Result:
        '''Schedule a timer which triggers every interval.'''
        if timer is None:
            timer = str(uuid7().uuid7)
        elif (task := self.timers.get(timer)) is not None:
            task.cancel()

        if trigger_at is None:
            tat = datetime.now(timezone.utc)
        else:
            tat = datetime.fromisoformat(trigger_at)
        
        if delay is not None:
            tat += timedelta(seconds=delay)
        
        self.local[timer] = {
            "extra": extra,
            "trigger_at": tat.isoformat(),
            "interval": interval
        }

        self.tg.create_task(self.task(timer, tat, interval))

        return {"timer": timer}
    
    @action
    async def cancel(self, *, timer: str, **_) -> Done:
        '''Cancel a timer.'''

        if (task := self.timers.get(timer)) is not None:
            task.cancel()
        del self.local[timer]

        return {"done": True}
    
    @action
    async def sleep(self, *,
            until: iso8601|None = None,
            delay: int|None = None,
            **_
        ) -> Done:
        '''Sleep until a datetime with a delay.'''
        now = datetime.now()
        ut = now if until is None else datetime.fromisoformat(until)
        if delay is not None:
            ut += timedelta(seconds=delay)
        await asyncio.sleep((ut - now).total_seconds())
        return {"done": True}

    @override
    async def bootstrap(self):
        # Restart all the existing timers
        for timer, state in self.local.items():
            self.timers[timer] = self.tg.create_task(self.task(
                timer,
                datetime.fromisoformat(state['trigger_at']),
                state['interval']
            ))
        
        while True:
            timer = await self.queue.get()
            if (state := self.local.get(timer)) is None:
                continue
            
            yield "trigger", state['extra'], {"timer": timer}