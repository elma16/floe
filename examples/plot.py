import sys
from seaice import *
from firedrake import *
from time import time
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

"""
Plotting script to plot the netcdf file if the model has diverged before the end of the timescale
"""

path = "./output/bt-adv-exp"
Path(path).mkdir(parents=True, exist_ok=True)

timescale = 604800
timestep = 1
timestepping = TimesteppingParameters(timescale=timescale, timestep=timestep)

title = "EVP Box Test Advected (timestep = 1 week)"
dataset_dirname = path + "/box_test_energy_T={}_t={}.nc".format(timescale, timestep)
plot_dirname = path + "/box_test_energy_T={}_t={}.png".format(timescale, timestep)

diagnostic = "energy"
dataset = Dataset(dataset_dirname, mode="r")
yaxis = dataset.variables[diagnostic][:]
dataset.close()

t = np.arange(0, len(yaxis) * timestep, timestep)

plt.plot(t, yaxis, label="timestep = {} [s]".format(timestep))
plt.ylabel(r"{} of solution".format(diagnostic))
plt.xlabel(r"Time [s]")
plt.title(title)
plt.legend(loc="best")
plt.savefig(plot_dirname)
