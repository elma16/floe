from firedrake import *
from netCDF4 import Dataset

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
    def __init__(self, variable):
        super().__init__(variable)

    def X_component(self):
        return 0

    def Y_component(self):
        return 0

    def Max_component(self):
        all_u, mesh, v_exp, zeta = model.sp_output()
        # projecting the solutions of the problem onto 'DG1'
        W = VectorFunctionSpace(mesh, "DG", 1)
        p = project(variable, W).dat.data
        # print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])
        return all_u


def korn_ineq(model):
    """Illustrating the failure of CR1 in Korn's Inequality"""
    print([norm(grad(model.data['velocity'][i])) > sqrt(
        norm(grad(model.data['velocity'][i]) + transpose(grad(model.data['velocity'][i])))) for i in
           range(len(model.data['velocity'][i]))])
