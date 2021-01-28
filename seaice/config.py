from firedrake import Constant

__all__ = ["OutputParameters", "TimesteppingParameters", "SeaIceParameters", "SolverParameters"]


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class OutputParameters(Configuration):
    """
    Output parameters
    """

    dump_vtus = True
    dumpfreq = 10
    dumplist = None
    dirname = None


class TimesteppingParameters(Configuration):
    """
    Timestepping parameters
    """
    timescale = None
    timestep = None
    alpha = 0.5
    maxk = 4
    maxi = 1


class SeaIceParameters(Configuration):
    """
    Physical parameters for the Sea Ice Momentum equations
    """

    rho = Constant(900)  # sea ice density (kg/m^3)
    cor = Constant(1.46 * 10 ** (-4))  # Coriolis parameter (-)
    rho_a = Constant(1.3)  # air density (kg/m^3)
    C_a = Constant(1.2 * 10 ** (-3))  # air drag coefficient (-)
    rho_w = Constant(1026)  # water density (kg/m^3)
    C_w = Constant(5.5 * 10 ** (-3))  # water drag coefficient (-)
    P_star = Constant(27.5 * 10 ** 3)  # ice strength parameter (N/m^2)
    C = Constant(20)  # ice concentration parameter (-)
    e = Constant(2)  # ellipse ratio (-)
    Delta_min = Constant(2 * 10 ** (-9))  # Lower bound for Delta (1/s)
    T = 100  # tuning parameter (s)
    alpha = Constant(500)  # mEVP constants
    beta = Constant(500)  # mEVP constants
    a_vp = 0.5 * 10 ** (-5)  # stabilisation constants
    a_evp = 0.1  # stabilisation constants
    a_mevp = 0.01  # stabilisation constants
    d = 1  # stabilisation constants


class SolverParameters(Configuration):
    """Solver Parameters"""
    srt_params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu"}
    bt_params = {"ksp_monitor": None, "snes_monitor": None, "ksp_type": "preonly", "pc_type": "lu", 'mat_type': 'aij'}
