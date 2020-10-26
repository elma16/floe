import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tests.strain_rate_tensor import *
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

'''
Creating all the vector plots and plotting error against time.
'''
#for k in [10,100,1000]:
#    t = np.arange(0,k,10)
#    plt.semilogy(t,strain_rate_tensor(k,timestep=10))
#    plt.ylabel(r'Error of solution ')
#    plt.xlabel(r'Time [s]')
#    plt.title(r'Error of computed solution for Section 4.1 Test, k = {}, T = {}'.format(k,k))
#    plt.show()
#    plt.savefig('./plots/strain_rate_velo_t={y}.png'.format(y=k))

#plotting convergence of velocity with timestep fixed, and mesh size changing.
for N in [10,20,30,50,100,200]:
    t = np.arange(0,10,10**(-1))
    plt.semilogy(t,strain_rate_tensor(T=10,timestep=10**(-1),number_of_triangles=N),label = 'mesh = %s' % N)
    plt.ylabel(r'Error of solution')
    plt.xlabel(r'Time [s]')
    plt.title(r'Error of computed solution for Section 4.1 Test')
    plt.legend(loc='best')
plt.show()
plt.savefig('./plots/strain_rate_veloN')

#all_errors1 = strain_rate_tensor(10**(-2),10**(-4))
#all_errors2 = strain_rate_tensor(10**(-2),10**(-5))
#all_errors3 = strain_rate_tensor(10**(-2),10**(-4),stabilised=2)
#all_errors4 = strain_rate_tensor(10**(-2),10**(-5),stabilised=2)

#t = np.arange(0, 10**(-2), 10**(-4))
#plt.plot(t, all_errors1,'r--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
#plt.plot(t,all_errors2,'b.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v + \nabla v^T)$')
#plt.plot(t,all_errors3,'g--',label = r'$n = 100, \sigma = \frac{\zeta}{2}(\nabla v)$')
#plt.plot(t,all_errors4,'k.',label = r'$n = 10, \sigma = \frac{\zeta}{2}(\nabla v)$')

#plt.legend(loc='best')
#plt.show()
