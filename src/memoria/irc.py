'''
IRC Interpreter for Memoria.

Connects to an IRC server, selectively responds to incoming messages, and
enters new memories into the subject including linking metadata concerning
the server, channel, and user that sourced them.
'''

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, override
from uuid import UUID

from pydantic import BaseModel
from pydantic_ai import Agent
from uuid_extension import uuid7

from aioipfs import AsyncIPFS
import bottom

class Channel(BaseModel):
    name: str
    key: str | None = None

class ServerState(BaseModel):
    nick: str
    host: str
    port: int
    channels: list[Channel]
    connected: bool

class ConnectResult(BaseModel):
    server: UUID

class Success(BaseModel):
    success: Literal[True]

class Error(BaseModel):
    error: str

class Server:
    '''Interpreter that connects to IRC and stores messages as memories.'''
    
    actions: tuple[str, ...] = ('connect', 'disconnect', 'recv', 'send')

    clients: dict[UUID, bottom.Client]

    def __init__(self, state: dict[UUID, ServerState]):
        super().__init__()
        self.state = state
        self.clients = {}

    async def connect(self, *, server: UUID, host: str, port: int) -> ConnectResult | Error:
        if (state := self.state.get(server)) is not None:
            if state.connected:
                return ConnectResult(server=server)
        else:
            self.state[server] = state = ServerState(host=host, port=port)
        state.host = host
        state.port = port
        state.connected = False
        irc = bottom.Client(host, port)
        try:
            await irc.connect()
            self.clients[server] = irc
            state.connected = True
            return ConnectResult(server=server)
        except Exception as e:
            return Error(error=e.args[0])
    
    async def disconnect(self, *, server: UUID) -> Success | Error:
        

    connect { server: UUID, url: string, port: int } => { server: UUID }
    connect { server: UUID, url: string, port: int } => { error: string }
    disconnect { server: UUID } => { success: true }
    disconnect { server: UUID } => { error: string }

    async def setup(self, irc: bottom.Client, state: ServerState):
        @irc.on('CLIENT_CONNECT')
        async def on_connect(**kwargs):
            await irc.send('NICK', nick=state.nick)
            await irc.send('USER', user=state.nick, realname='Memoria IRC Bot')
        
        @irc.on('CLIENT_DISCONNECT')
        async def on_disconnect(**kwargs):
            print("Disconnected from IRC server")
        
        @irc.on('RPL_WELCOME')
        async def on_welcome(**kwargs):
            # Join channels after successful connection
            for chan in state.channels:
                await irc.send('JOIN', channel=chan.name, key=chan.key)
        
        @irc.on('PRIVMSG')
        async def on_privmsg(nick, target, message, **kwargs):
            # Don't process our own messages
            if nick == state.nick:
                return
            await self._handle_message(target, nick, message)
        
        await irc.connect()
    
    @override
    async def recv(self, update: 'MemoryDAG'):
        roots = list(update)
        if self.last_message:
            roots.append(self.last_message)
        g = self.subject.recall(list(update))
        res = await self.process(g)

    async def send_message(self, channel: str, message: str):
        '''Send a PRIVMSG to a channel or user.'''
        await irc.send('PRIVMSG', target=channel, message=message)

    async def _handle_message(self, channel: str, nick: str, message: str):
        '''
        Handle an incoming IRC message.
        Creates memories and optionally responds.
        '''

        m: Memory
        
        # Ensure channel CID
        if (channel_cid := self.channels_cids.get(channel)) is None:
            # Ensure server CID
            if (server_cid := self.server_cid) is None:
                m = Memory(
                    data=MetaData(metadata={"server": self.server_name}),
                    edges=set()
                )
                self.server_cid = server_cid = m.cid
                self.pending_updates.append(m)
            
            m = Memory(
                data=MetaData(metadata={"channel": channel}),
                edges={server_cid}
            )
            self.channels_cids[channel] = channel_cid = m.cid
            self.pending_updates.append(m)
        
        # Ensure user CID
        if (user_cid := self.users_cids.get(nick)) is None:
            m = Memory(
                data=MetaData(metadata={"user": nick}),
                edges=set()
            )
            self.users_cids[nick] = user_cid = m.cid
            self.pending_updates.append(m)
        
        edges = {channel_cid, user_cid}
        if (last_message := self.last_message) is not None:
            edges.add(last_message)

        # Create message memory
        m = Memory(
            uuid=uuid7().uuid7,
            data=OtherData(parts=[TextPart(content=message)]),
            edges=edges
        )
        msg_cid = m.cid
        self.last_message = msg_cid
        self.pending_updates.append(m)
    
    async def bootstrap(self):
        '''Main event loop for processing IRC messages.'''

        for uid, state in self.state.items():
            irc = bottom.Client(state['host'], state['port'])

            await self.setup(irc)
            self.clients[uid] = irc
            
            # Keep the connection alive
            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                pass

async def main():
    '''Main entry point for the IRC interpreter.'''
    import sys
    
    # Default configuration - can be overridden via command line or config file
    subject_url = "http://localhost:8000"
    server = "irc.rizon.net"
    port = 6667
    nickname = "memoria"
    channels = ["#meta"]
    
    # Simple argument parsing
    if len(sys.argv) > 1:
        subject_url = sys.argv[1]
    if len(sys.argv) > 2:
        server = sys.argv[2]
    if len(sys.argv) > 3:
        port = int(sys.argv[3])
    if len(sys.argv) > 4:
        nickname = sys.argv[4]
    if len(sys.argv) > 5:
        channels = sys.argv[5].split(',')
    
    irc = bottom.Client(server, port)
    with database() as db:
        subject = Subject(db)
        async with AsyncIPFS() as ipfs:
            agent = Agent("gpt-4o-mini")
            interpreter = IRCInterpreter(
                irc, subject, ipfs, server, nickname, channels
            )
            try:
                await interpreter.run()
            except KeyboardInterrupt:
                print("\nShutting down IRC interpreter...")   

if __name__ == "__main__":
    asyncio.run(main())

