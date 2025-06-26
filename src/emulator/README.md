An Emulator in Memoria is a system equipped with one or more LLMs which can advance the memoria state with external grounding.

- `Emulator` provides the common interface for emulators.
- `ServerEmulator` is an emulator with direct access to memoria's state for use in servers.
- `ClientEmulator` is an emulator that communicates with a server emulator over MCP.