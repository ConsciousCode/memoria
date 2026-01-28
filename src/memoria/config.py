'''
General config within the Memoria system.
'''

from pydantic import BaseModel

class RecallConfig(BaseModel):
    '''Configuration for memory recall.'''
    
    refs: int = 5
    '''Maximum depth of memories after roots.'''
    deps: int = 5
    '''Maximum depth of memories before roots.'''
    memories: int = 5
    '''Maximum number of memories.'''