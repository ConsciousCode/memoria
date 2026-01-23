from typing import Literal, NotRequired, TypedDict, override

from memoria.hypersync import Concept, action, event

class StaticState(TypedDict):
    done: NotRequired[bool]

class Bootstrap(Concept[object, StaticState]):
    '''
    Concept used to bootstrap the rest of the system on the first boot.
    '''
    
    class Done(TypedDict):
        done: Literal[True]

    @event
    async def once(self, **_) -> Done:
        '''
        An event which is called exactly once when a concept is loaded across
        all boots.
        '''
        self.static['done'] = True
        return {"done": True}
    
    @action
    async def reset(self, **_) -> Done:
        '''
        Reset the concept to activate again on the next boot.
        '''
        self.static['done'] = False
        return {"done": True}

    @override
    async def bootstrap(self):
        if self.static.get("done"):
            return
        
        yield "once", {}, None