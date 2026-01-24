from typing import Literal, TypedDict

from memoria.hypersync import ActionId, Bindings, Concept, action, value_t

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

    @action
    async def ignored(self, *,
            action: ActionId,
            params: Bindings,
            result: Bindings
        ):
        '''An action completion was not matched with any syncs.'''
        return {}

    @action
    async def event_invoked(self, *,
            event: ActionId,
            params: Bindings
        ):
        '''An event was invoked as if it was an action.'''
        return {}

    @action
    async def no_action(self, *,
            action: ActionId,
            params: Bindings
        ):
        '''A nonexistent action was invoked.'''
        return {}

    @action
    async def concept(self, *,
            concept: str,
            params: Bindings
        ):
        '''A nonexistent concept was referenced.'''
        return {}