from firedrake import dot
from abc import ABCMeta, abstractmethod, property


class DiagnosticField(object, metaclass=ABCMeta):

    def __init__(self, required_fields=()):
        self._initialised = False
        self.required_fields = required_fields

    @property
    def name(self):
        """The name of this diagnostic field"""
        pass

    def setup(self, state, space=None):
        if not self._initialised:
            if space is None:
                space = state.spaces("DG0", state.mesh, "DG", 0)
            self.field = state.fields(self.name, space, pickup=False)
            self._initialised = True

    @abstractmethod
    def compute(self, state):
        """ Compute the diagnostic field from the current state"""
        pass

    def __call__(self, state):
        return self.compute(state)


class Energy(DiagnosticField):

    def kinetic(self, u):
        return 0.5 * dot(u, u)
