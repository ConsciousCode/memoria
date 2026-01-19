import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Literal, NotRequired, Protocol, TypedDict, cast, override
from uuid import UUID
from dataclasses import dataclass
import calendar
import re

from uuid_extension import uuid7
from memoria.hypersync import Bindings, Concept, action, event, mutable_t, value_t

type iso8601 = str

SIMPLE = re.compile(r'(?:\*|(\d+|\w+)(?:-(\d+|\w+))?)(?:/(\d+))')
SPECIAL_DOW = re.compile(r'(\d+)(#[1-5]|L)')

@dataclass(frozen=True)
class SpecialDOM:
    kind: Literal["L", "W", "LW"]
    base: int
    
    @classmethod
    def parse(cls, s: str):
        if s == "L" or s == "LW":
            return cls(s, 0)
        if s.endswith("W"):
            return cls("W", int(s[:-1]))
        raise ValueError(r"Could not parse field {s!r}")

    def match(self, date: date):
        if "L" in self.kind:
            _, base_day = calendar.monthrange(date.year, date.month)
        else:
            base_day = self.base
        
        if "W" in self.kind:
            # Not a weekday, cannot match
            if date.weekday() > calendar.FRIDAY:
                return False
            
            td = date.replace(day=base_day)
            if td.weekday() == calendar.SATURDAY:
                # Nearest is Friday
                if td - timedelta(days=1) == date:
                    return True
            if td.weekday() == calendar.SUNDAY:
                # Nearest is Monday
                if td + timedelta(days=1) == date:
                    return True
        elif date.day == base_day:
            return True
        return False

@dataclass(frozen=True)
class SpecialDOW:
    kind: Literal["L", "#"]
    week: int
    nth: int
    
    @classmethod
    def parse(cls, s: str):
        if m := SPECIAL_DOW.match(s):
            w, n = m.groups()
            w = int(w)
            if n[0] == "L":
                return cls("L", w, 0)
            else:
                # Make it 0-indexed for later nth offset
                return cls("#", w, int(n[1:]) - 1)
        raise ValueError(f"Could not parse field {s!r}")

    def match(self, date: date):
        if self.kind == "L":
            # Find the last day of week
            _, last = calendar.monthrange(date.year, date.month)
            base = date.replace(day=last)
            return date == base - timedelta(
                days=base.weekday() - (calendar.SUNDAY - self.week)
            )
        else:
            # Find the first day of week plus the nth offset
            base = date.replace(day=1)
            return date == base + timedelta(
                days=base.weekday() + (calendar.SUNDAY - self.week) + 7*self.nth
            )

class CronField[T: SpecialDOM | SpecialDOW]:
    simple: set[int]
    special: list[T]
    
    def __init__(self, fields: Iterable[int | T]):
        simple = set()
        special = []
        for f in fields:
            if isinstance(f, int):
                simple.add(f)
            else:
                special.append(f)
        self.simple = simple
        self.special = special

    def match(self, time: datetime):
        if time.month in self.simple:
            return True
        
        date = time.date()
        return any(special.match(date) for special in self.special)

def _parse_field[T](
        field: str,
        lo: int, hi: int,
        parse: Callable[[str], T] | None = None,
        names: tuple[str, ...] = ()
    ) -> Iterable[int | T]:
    def simple(s: str):
        if s in names:
            return names.index(s) + lo
        return int(s)
    
    if field == "*":
        yield from range(lo, hi + 1)
        return
    
    for part in field.split(','):
        if (m := SIMPLE.match(part)) is None:
            if parse is None:
                raise ValueError(f"Could not parse field {part!r}")
            yield parse(part)
            continue

        s, e, p = m.groups()
        if s is None:
            # Wildcard *
            start, end = lo, hi
        elif e is None:
            # Single value
            start = end = simple(s)
        else:
            # Range
            start = simple(s)
            end = simple(e)
        
        step = 1 if p is None else int(p)

        if start < lo:
            raise ValueError(f"Start of range {start} below min {lo}")
        if end > hi:
            raise ValueError(f"End of range {end} above max {lo}")
        if end < start:
            raise ValueError(f"End of range {end} < start of range {start}")
        if step <= 0:
            raise ValueError(f"Invalid range step size {step}")

        yield from range(start, end + 1, step)

@dataclass(frozen=True)
class Entry:
    name: str|None

    minute: set[int]
    hour: set[int]
    month: CronField[SpecialDOM]
    day: set[int]
    week: CronField[SpecialDOW]

    @classmethod
    def parse(cls, name: str|None, entry: str):
        match entry:
            case "@yearly"|"@annually":
                entry = "0 0 1 1 *"
            case "@monthly":
                entry = "0 0 1 * *"
            case "@weekly":
                entry = "0 0 * * 0"
            case "@daily"|"@midnight":
                entry = "0 0 * * *"
            case "@hourly":
                entry = "0 * * * *"
        
        i, h, d, m, w = entry.split(' ')
        return cls(
            name,
            set(_parse_field(i, 0, 59)),
            set(_parse_field(h, 0, 23)),
            CronField(_parse_field(d, 1, 31, SpecialDOM.parse)),
            set(_parse_field(m, 1, 12, names=(
                "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
            ))),
            CronField(_parse_field(
                w, 0, 7, SpecialDOW.parse, (
                    "SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"
                )
            ))
        )
    
    def match(self, time: datetime):
        if time.minute not in self.minute:
            return False
        if time.hour not in self.hour:
            return False
        if self.month.match(time):
            return False
        if self.day and self.week:
            # Match occurs if either field matches
            if time.day not in self.day and self.week.match(time):
                return False
        return True

class Cron(Concept):
    """Source of time-based events"""

    tg: asyncio.TaskGroup
    dirty: set[UUID]

    def __init__(self):
        super().__init__()
        self.tg = asyncio.TaskGroup()
        self.dirty = set()
    
    async def __aenter__(self):
        await self.tg.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        return await self.tg.__aexit__(exc_type, exc, tb)
    
    def normalize(self, this: str|None):
        if this is None:
            timer = uuid7().uuid7
        else:
            timer = UUID(this)
            self.dirty.add(timer)
        
        return timer
    
    class Now(TypedDict):
        datetime: iso8601

    @action
    async def now(self) -> Now:
        return {"datetime": datetime.now(timezone.utc).isoformat()}
    
    @event
    async def job(self, *, name: str|None = None, cron: str):
        '''Event for when a cron entry is matched.'''
    
    @event
    async def reboot(self):
        '''Event for when the cron concept starts.'''

    class Construct(TypedDict):
        cron: str
    
    @action
    async def schedule(self, *,
            cron: str | None,
            name: str | None = None,
            entry: str
        ) -> Construct:
        '''Schedule a new cron job.'''
        this = self.normalize(cron)
        state: dict[str, mutable_t] = {"entry": entry}
        if name is not None:
            state['name'] = name
        self.state[this] = state

        return {"cron": str(this)}
    
    class Success(TypedDict):
        success: bool

    @action
    async def unschedule(self, *,
            cron: str
        ) -> Success:
        '''Unschedule an existing cron job.'''
        this = UUID(cron)
        if this not in self.state:
            raise LookupError(f"cron job {this}")
        
        del self.state[this]
        self.dirty.add(this)

        return {"success": True}
    
    @override
    async def bootstrap(self):
        yield "reboot", {}, {}

        tab = dict[UUID, Entry]()
        while True:
            for dirty in self.dirty:
                state = self.state[dirty]
                if isinstance(st := state.get('entry'), str):
                    if isinstance(name := state.get('name'), str|None):
                        tab[dirty] = Entry.parse(name, st)
                else:
                    del tab[dirty]
            
            now = datetime.now()
            for cron, entry in tab.items():
                if not entry.match(now):
                    continue
                
                result: dict[str, value_t] = {"cron": str(cron)}
                if (name := entry.name) is not None:
                    result['name'] = name

                yield "job", {}, result
            
            # Sleep until the next minute boundary
            later = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            await asyncio.sleep((later - now).total_seconds())