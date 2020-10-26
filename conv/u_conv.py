import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.strain_rate_tensor import *
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

'''
Creating all the vector plots and plotting error against time.
'''


starttime = time.time()
all_errors1 = strain_rate_tensor(10**(-2),10**(-4))
all_errors2 = strain_rate_tensor(10**(-2),10**(-4),number_of_triangles=10)
all_errors3 = strain_rate_tensor(10**(-2),10**(-4),stabilised=True)
all_errors4 = strain_rate_tensor(10**(-2),10**(-4),stabilised=True,number_of_triangles=10)
endtime = time.time()
print(endtime-starttime)

length_of_time = 10**(-2)
timestep = 10**(-4)
t = np.arange(0, length_of_time, timestep)
plt.plot(t, all_errors1,'r--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
plt.plot(t,all_errors2,'b.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
plt.plot(t,all_errors3,'g--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v)$')
plt.plot(t,all_errors4,'k.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v)$')
plt.ylabel(r'Error of solution $[\times 10^3]$')
plt.xlabel(r'Time [s]')
plt.title(r'Error of computed solution for Section 4.1 Test, $k = 10^{-4}, T = 10^{-2}$')
plt.legend(loc='best')
plt.show()