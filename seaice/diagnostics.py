from firedrake import *
from netCDF4 import Dataset
import time
import numpy as np


__all__ = ["Error", "Energy", "Velocity", "OutputDiagnostics"]


class Diagnostic(object):
    def __init__(self, v):
        self.v = v


class Error(Diagnostic):

    def __init__(self, v, solution):
        super().__init__(v)
        self.solution = solution

    @staticmethod
    def compute(v, solution):
        return norm(solution - v)


class Energy(Diagnostic):
    def __init__(self, v):
        super().__init__(v)

    @staticmethod
    def compute(v):
        return assemble(inner(grad(v), grad(v)) * dx)


class Velocity(Diagnostic):
    def __init__(self, v, mesh):
        super().__init__(v)
        self.mesh = mesh

    def x_component(fs,v):
        vh =  v[0]
        return interpolate(vh,fs)

    @staticmethod
    def y_component(v):
        return v[1]

    @staticmethod
    def max_component(v,mesh):
        W = VectorFunctionSpace(mesh, "DG", 1)
        p = project(v, W).dat.data
        return p
    
# currently only works for the diagnostics of one model in one file
class OutputDiagnostics(object):
    """
    creates a netCDF file with all the diagnostic data
    """

    def __init__(self, dirname, description):
        self.dirname = dirname
        self.description = description

        with Dataset(dirname, "w") as dataset:
            dataset.description = "Diagnostics data for simulation {desc}".format(desc=description)
            dataset.history = "Created {t}".format(t=time.ctime())
            dataset.source = "Output from SeaIce Model"
            dataset.createDimension("time", None)
            times = dataset.createVariable("time", np.float64, ("time",))
            times.units = "seconds"
            dataset.createVariable("energy", np.float64, ("time",))
            dataset.createVariable("error", np.float64, ("time",))

    def dump(self, variable, t, solution=None):
        with Dataset(self.dirname, "a") as dataset:
            idx = dataset.dimensions["time"].size
            dataset.variables["time"][idx:idx + 1] = t
            energy = dataset.variables["energy"]
            error = dataset.variables["error"]
            energy[idx:idx + 1] = Energy.compute(variable)
            if solution is not None:
                error[idx:idx + 1] = Error.compute(variable, solution)
