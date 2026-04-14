import threading

from .core import AVPlumber

class NodeBase():
    def __init__(self, avplumber: AVPlumber, args: dict):
        self._avplumber = avplumber
        self._args = args

    def process(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class PythonNode(NodeBase):
    def __init__(self, avplumber: AVPlumber, args: dict):
        super().__init__(avplumber, args)
        self.__thread = threading.Thread(target=self.__thread_function)
        self.__running = False

    def __thread_function(self):
        while self.__running:
            self.process()

    def process(self):
        raise NotImplementedError("process() must be implemented in subclass")

    def start(self):
        self.__running = True
        self.__thread.start()

    def stop(self):
        self.__running = False
        self.__thread.join()


class PythonNodeSingleOutput(PythonNode):
    def __init__(self, avplumber: AVPlumber, args: dict):
        super().__init__(avplumber, args)
        self._dst = avplumber.getEdge(args["dst"])
        assert self._dst is not None, "Destination edge not found"


class PythonNodeSingleInput(PythonNode):
    def __init__(self, avplumber: AVPlumber, args: dict):
        super().__init__(avplumber, args)
        self._src = avplumber.getEdge(args["src"])
        assert self._src is not None, "Source edge not found"


class PythonNodeSISO(PythonNodeSingleInput, PythonNodeSingleOutput):
    def __init__(self, avplumber: AVPlumber, args: dict):
        super().__init__(avplumber, args)
