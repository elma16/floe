from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from seaice.models import *

# TODO Get the Error diagnostic to the point in which you can plot stuff with it
# TODO : get component of UFL velocity


__all__ = ["Error", "Energy", "Velocity"]


class Diagnostic(object):
    def __init__(self, model):
        if not isinstance(model, SeaIceModel):
            raise RuntimeError("You must use a sea ice model")
        else:
            self.model = model

    @staticmethod
    def max():
        return 0

    @staticmethod
    def min():
        return 0


class Error(Diagnostic):

    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def compute(model):
        return [errornorm(model.v_exp, model.data['velocity'][i]) for i in range(len(model.data['velocity']))]


class Energy(Diagnostic):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def compute(model):
        return [norm(model.data['velocity'][i], norm_type="H1") for i in range(len(model.data['velocity']))]


class Velocity(Diagnostic):
    def __init__(self, model):
        super().__init__(model)

    def X_component(self):
        return 0

    def Y_component(self):
        return 0

    def Max_component(self):
        all_u, mesh, v_exp, zeta = model.sp_output()
        # projecting the solutions of the problem onto 'DG1'
        W = VectorFunctionSpace(mesh, "DG", 1)
        p = [project(model.data['velocity'][i], W).dat.data for i in range(len(model.data['velocity']))]
        print(shape(p[0]))
        # print([all_u[i].evaluate((,),'x',0,0) for i in range(len(all_u))])
        return all_u


def korn_ineq(model):
    """Illustrating the failure of CR1 in Korn's Inequality"""
    print([norm(grad(model.data['velocity'][i])) > sqrt(
        norm(grad(model.data['velocity'][i]) + transpose(grad(model.data['velocity'][i])))) for i in
           range(len(model.data['velocity'][i]))])
