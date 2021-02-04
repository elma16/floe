from netCDF4 import Dataset
import numpy as np
import time

filename = "./output/test.nc"
description = "hello"

with Dataset(filename, "w") as dataset:
    dataset.description = "Diagnostics data for simulation {desc}".format(desc=description)
    dataset.history = "Created {t}".format(t=time.ctime())
    dataset.source = "Output from SeaIce Model"
    dataset.createDimension("time", None)
    times = dataset.createVariable("time", np.float64, ("time",))
    times.units = "seconds"
    dataset.createVariable("energy", np.float64, ("time",))
    dataset.createVariable("error", np.float64, ("time",))

with Dataset(filename, "a") as dataset:
    energy[:] = [1, 2, 3]
    print(dataset)
    # idx = dataset.dimensions["time"].size
    # dataset.variables["time"][idx:idx + 1] = 0
    # times[idx:idx+1] = 0

# longitudes[:] = lons

# print(latitudes[:])
