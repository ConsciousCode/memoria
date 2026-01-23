from typing import TypedDict

from memoria.hypersync import Concept, action, value_t

class Report(Concept):
    '''Concept for reporting unusual conditions.'''

    class Echo(TypedDict):
        error: value_t
    
    @action
    async def uncaught(self, *, error: value_t, **_) -> Echo:
        '''
        An uncaught error was detected. No-op on its own, it's intended to
        give syncs something to match.
        '''
        return {"error": error}