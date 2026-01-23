import asyncio
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Literal, TypedDict, override
from dataclasses import dataclass
import calendar
import re

from uuid_extension import uuid7

from memoria.hypersync import Bindings, Concept, FlowId, action, event

type iso8601 = str

SIMPLE = re.compile(r'(?:\*|(\d+|\w+)(?:-(\d+|\w+))?)(?:/(\d+))')
SPECIAL_DOW = re.compile(r'(\d+)(#[1-5]|L)')

@dataclass(frozen=True)
class SpecialDOM:
    '''Day Of the Month special value.'''

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
    '''Day Of the Week special value.'''

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
    '''Cron field supporting special values.'''
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
    '''Parse a cron field.'''

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
class Spec:
    '''Entry in the crontab.'''

    minute: set[int]
    hour: set[int]
    month: CronField[SpecialDOM]
    day: set[int]
    week: CronField[SpecialDOW]

    @classmethod
    def parse(cls, entry: str):
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

class LocalState(TypedDict):
    extra: Bindings
    spec: str

class Cron(Concept[LocalState]):
    """
    Cron concept implementation which generates `job` events whenever an entry
    in the crontab is matched. Supports 5-field L/W in DOM and L/# in DOW.
    """

    tab: dict[str, Spec]

    def __init__(self):
        super().__init__()
        self.tab = {}
    
    class Result(TypedDict):
        cron: str
    
    class Done(TypedDict):
        done: Literal[True]

    @event
    async def job(self, **_) -> Result:
        '''Event for when a cron entry is matched.'''
        ...
    
    @event
    async def reboot(self, **_) -> Result:
        '''Event for when the cron concept starts.'''
        ...
    
    @action
    async def schedule(self, *,
            flow: FlowId,
            cron: str|None,
            spec: str,
            **extra
        ) -> Result:
        '''Schedule a new cron job.'''

        if cron is None:
            cron = str(uuid7().uuid7)
        self.tab[cron] = Spec.parse(spec)
        self.state[cron] = {
            "extra": extra,
            "spec": spec
        }
        return {"cron": cron}

    @action
    async def cancel(self, *, cron: str, **_) -> Done:
        '''Cancel an existing cron job.'''
        if cron not in self.state:
            raise LookupError(f"cron job {cron}")
        
        del self.tab[cron]
        del self.state[cron]

        return {"done": True}
    
    @override
    async def bootstrap(self):
        yield "reboot", {}, {}

        for cron, state in self.local.items():
            self.tab[cron] = Spec.parse(state['spec'])

        while True:
            now = datetime.now()
            for cron, spec in self.tab.items():
                if spec.match(now):
                    state = self.local[cron]
                    yield "job", state['extra'], {"cron": cron}
            
            # Sleep until the next minute boundary
            later = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            await asyncio.sleep((later - now).total_seconds())