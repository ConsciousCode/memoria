#!/usr/bin/env python3

'''
CLI for interacting with Memoria systems, particularly via the Memoria Subject.
'''

import asyncio
import re
from typing import Mapping, overload, override

import aioconsole

from cid import CID
from memoria.concepts.bootstrap import Bootstrap
from memoria.concepts.cron import Cron
from memoria.concepts.report import Report
from memoria.concepts.http import HTTP
from memoria.concepts.test import Test
from memoria.concepts.timer import Timer
from memoria.concepts.watchdog import Watchdog
from memoria.hypersync import ActionId, Bindings, Completion, Engine, FlowId, Sync, Then, Var, When, value_t

ACTION = re.compile(r'''([^/]+)/(\w[\w\d]*)''')
SPACE = re.compile(r'\s+')
KEY = re.compile(r'''
    (?P<unq>\w[\w\d+]*)       |
    "(?P<quo>(?:[^"]+|\\.)*)" |
    \?(?P<var>\w[\w\d]*)
''', re.X)
VALUE = re.compile(r'''
    \?(?P<var>\w[\w\d]*)          |
    "(?P<str>[^"]+|\\\\.)"        |
    (?P<real>\d+\.\d* | \d*\.\d+) |
    (?P<int>\d+)                  |
    (?P<bool>true|false)          |
    (?P<null>null)                |
    (?P<obj>\{)                   |
    (?P<arr>\[)
''', re.X)

QUOT_T = str.maketrans({
    "\\": "\\",
    '"': '"'
})

def unescape(s: str):
    return re.sub(r"\\(.)", lambda m: m[1].translate(QUOT_T), s)

class QueryParser:
    def __init__(self, s: str, env: Bindings):
        self.s = s.strip()
        self.pos = 0
        self.env = env
    
    @overload
    def maybe(self, r: re.Pattern[str]) -> re.Match[str]|None: ...
    @overload
    def maybe(self, r: str) -> str|None: ...

    def maybe(self, r: re.Pattern[str]|str):
        if m := SPACE.match(self.s, self.pos):
            self.pos += len(m[0])

        if isinstance(r, str):
            if self.s.startswith(r, self.pos):
                self.pos += len(r)
                return r
            return None

        if m := r.match(self.s, self.pos):
            self.pos += len(m[0])
        return m
    
    @overload
    def expect(self, name: str, r: re.Pattern[str]) -> re.Match[str]: ...
    @overload
    def expect(self, name: str, r: str) -> str: ...

    def expect(self, name: str, r: re.Pattern[str]|str):
        if m := self.maybe(r):
            return m
        print(self.s[self.pos:])
        raise ValueError(f"Expected {name}")

    def value(self):
        m = self.expect("value", VALUE)
        match m.lastgroup:
            case "var": return self.env[m['var']]
            case "str": return m['str']
            case "real": return float(m['float'])
            case "int": return int(m['int'])
            case "bool": return m['bool'] == "true"
            case "null": return None
            case "obj": return self.object()
            case "arr": return self.array()
            case _: assert False

    def object(self):
        obj: Bindings = {}
        if not self.maybe("}"):
            while True:
                k = self.expect("key", KEY)
                if self.maybe(":"):
                    value = self.value()
                    match k.lastgroup:
                        case "unq": key = str(k['unq'])
                        case "quo": key = unescape(k['quo'])
                        case "var":
                            key = self.env[k['var']]
                            if not isinstance(key, str):
                                raise TypeError("Variable key not a string")
                        case _: assert False
                    obj[key] = value
                elif k.lastgroup != "var":
                    raise ValueError("Shorthand key requires a var")
                else:
                    obj[k['var']] = self.env[k['var']]
                
                if self.maybe("}"):
                    break
                self.expect("comma", ",")
        
        return obj
    
    def array(self):
        arr: list[value_t] = []
        if not self.maybe("]"):
            while True:
                arr.append(self.value())
                if self.maybe("]"):
                    break
                self.expect("comma", ",")
        
        return arr
    
    def parse(self):
        act = self.expect("action", ACTION)
        self.expect("params", "{")
        params = self.object()
        if self.maybe("=>"):
            self.expect("result", "{")
            result = self.object()
        else:
            result = None
        
        return act[0], params, result

class CLIEngine(Engine):
    @override
    async def invoke(self,
            action: ActionId,
            params: Bindings,
            flow: FlowId |  None = None,
            trigger: CID | None = None
        ):
        print("invoke", action, params)
        return await super().invoke(action, params, flow, trigger)

    @override
    async def complete(self,
            action: ActionId,
            params: Bindings,
            result: Bindings,
            state: Mapping[str, Bindings] | None = None,
            flow: FlowId|None = None,
            trigger: CID|None = None
        ):
        print("complete", action, params, "=>", result)
        return await super().complete(action, params, result, state, flow, trigger)

    @override
    async def uncaught(self,
            error: Exception,
            flow: FlowId,
            trigger: CID | None = None
        ):
        print("Uncaught", error)
        return await super().uncaught(error, flow, trigger)
    
    @override
    async def ignored(self, cmp: Completion):
        print("Ignored:", cmp.action, cmp.params, cmp.result)
    
    @override
    async def event_invoked(self, event: ActionId):
        print("EventInvoked:", event)
    
    @override
    async def no_such_action(self, action: ActionId):
        print("NoSuchAction:", action)

    @override
    async def no_such_concept(self, concept: str):
        print("NoSuchConcept:", concept)

async def repl(engine: CLIEngine):
    env = {}
    while True:
        line: str
        if not (line := await aioconsole.ainput("> ")):
            continue
        
        if line[0] == '/':
            words = line[1:].split()
            match words[0]:
                case "": continue
                case "h" | "help":
                    print("Concept/action {...params} [=> {...result}]")
                    print("/h[elp]")
        else:
            act, params, result = QueryParser(line, env).parse()
            if result is not None:
                await engine.complete(act, params, result)
                continue

            if (result := await engine.invoke(act, params)) is None:
                # Invocation failed
                continue

            if (error := result.get("error")) is None:
                print(result)
                continue

            match error:
                case {"type": errt, "message": message, "traceback": list(tb)}:
                    print("Traceback (most recent call last):")
                    print(''.join(map(str, tb)))
                    print(f"{errt}: {message}")
                
                case _:
                    print("Error:", error)

async def main(name, *argv):
    engine = CLIEngine({},
        concepts=[
            Cron(),
            Report(),
            HTTP(),
            Timer(),
            Watchdog(),
            Test(),
            Bootstrap()
        ],
        syncs=[
            Sync("BootstrapHTTP",
                purpose="Bootstrap the HTTP server.",
                when=[
                    When("Bootstrap/once", {}, {})
                ],
                then=[
                    Then("HTTP/start", {"port": 8060})
                ]
            ),
            Sync("HTTPDummy",
                purpose="Dummy output for HTTP requests",
                when=[
                    When("HTTP/request", {"method": "GET"}, {"request": Var("request")})
                ],
                then=[
                    Then("HTTP/respond", {"request": Var("request"), "text": "yes"}),
                    Then("Stdio/print", {"data": "yes"})
                ]
            )
        ]
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(engine.run())
        tg.create_task(repl(engine))

if __name__ == "__main__":
    import sys
    asyncio.run(main(*sys.argv))
