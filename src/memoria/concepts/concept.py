from typing import Literal, TypedDict, override

from memoria.hypersync import ActionId, Bindings, Concept, Engine, FlowId, Sync, action, stimulus

HyperInvoke = Sync("HyperInvoke", 
    "Hypersync which invokes a "
)

class HyperConcept(Concept):
    '''
    Concept for manipulating and referring to concepts.
    '''

    name = "Concept"

    engine: Engine

    def load(self, engine: Engine):
        self.engine = engine

    @action
    async def invoke(self, *, action: ActionId, params: Bindings, flow: FlowId):
        '''Pivot for invoking actions.'''
        return {}
    
    class Done(TypedDict):
        done: Literal[True]

    @stimulus
    async def once(self) -> Done:
        '''
        An event which is called exactly once when a concept is loaded across
        all boots.
        '''
        self.static['done'] = True
        return {"done": True}
    
    @action
    async def reset(self) -> Done:
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