from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Plotter"]


class Plotter(object):
    def __init__(self, plot_dirname, title, dataset_dirname, diagnostic, timestepping):
        self.title = title
        self.plot_dirname = plot_dirname
        self.dataset_dirname = dataset_dirname
        self.diagnostic = diagnostic
        self.timestepping = timestepping
        self.timestep = timestepping.timestep
        self.timescale = timestepping.timescale

        dataset = Dataset(dataset_dirname, mode='r')
        self.yaxis = dataset.variables[diagnostic][:]
        dataset.close()

    def plot(self,plot_option='plot'):
        
        '''
        plot :: choose what plot you want to make
        '''

        t = np.arange(0, self.timescale, self.timestep)
        if plot_option == 'plot':
            plt.plot(t, self.yaxis, label="timescale = {}".format(self.timescale))
        elif plot_option == 'loglog':
            plt.loglog(t, self.yaxis, label="timescale = {}".format(self.timescale))
        elif plot_option == 'semilogy':
            plt.semilogy(t, self.yaxis, label="timescale = {}".format(self.timescale))
        elif plot_option == 'semilogx':
            plt.semilogx(t, self.yaxis, label="timescale = {}".format(self.timescale))
        plt.ylabel(r'{} of solution'.format(self.yaxis))
        plt.xlabel(r'Time [s]')
        plt.title(self.title)
        plt.legend(loc='best')
        plt.savefig(self.plot_dirname)
