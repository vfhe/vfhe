import asyncio


class IOPValue(asyncio.Future):
    def __init__(self, value=None, loop=None):
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
        super().__init__(loop=loop)
        if value is not None:
            self.set_result(value)


class IOPVariable(asyncio.Future):
    def __init__(self, name: str, value=None, loop=None):
        self.name = name
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
        super().__init__(loop=loop)
        if value is not None:
            self.set_result(value)


class IOPParty:
    def __init__(self, state=None):
        self.state = state


class IOPProver(IOPParty):
    def __init__(self, state=None):
        super().__init__(state=state)


class IOPVerifier(IOPParty):
    def __init__(self, state=None):
        super().__init__(state=state)
