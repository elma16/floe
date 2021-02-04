from firedrake import *
from netCDF4 import Dataset
import time
import numpy as np

# TODO Get the Error diagnostic to the point in which you can plot stuff with it
# TODO : get component of UFL velocity


__all__ = ["Error", "Energy", "Velocity"]


class Diagnostic(object):
    def __init__(self, variable):
        self.variable = variable


class Error(Diagnostic):

    def __init__(self, variable, solution):
        super().__init__(variable)
        self.solution = solution

    @staticmethod
    def compute(variable, solution):
        return norm(solution - variable)


class Energy(Diagnostic):
    def __init__(self, variable):
        super().__init__(variable)

    @staticmethod
    def compute(variable):
        return norm(variable, norm_type="H1")


class Velocity(Diagnostic):
    def __init__(self, variable, mesh):
        super().__init__(variable)
        self.mesh = mesh

    def X_component(self):
        return self.variable[0]

    def Y_component(self):
        return self.variable[1]

    def Max_component(self):
        # projecting the solutions of the problem onto 'DG1'
        W = VectorFunctionSpace(self.mesh, "DG", 1)
        p = project(variable, W).dat.data
        # print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])
        return p

    def korn_ineq(self):
        return norm(grad(self.variable)) > sqrt(norm(grad(self.variable) + transpose(grad(self.variable))))


class OutputDiagnostics(object):
    def __init__(self, dirname, variable, solution, description):
        self.dirname = dirname
        self.variable = variable
        self.solution = solution
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

    def dump(self, t):
        with Dataset(self.dirname, "a") as dataset:
            idx = dataset.dimensions["time"].size
            dataset.variables["time"][idx:idx + 1] = t
            energy = dataset.variables["energy"]
            error = dataset.variables["error"]
            energy[idx:idx + 1] = Energy.compute(self.variable)
            error[idx:idx + 1] = Error.compute(self.variable,self.solution)