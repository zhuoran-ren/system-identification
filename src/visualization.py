"""This class is used to virsualize the system identification result"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cmath
import scipy.signal as signal

class Visualization():
    """Visualize the system identification result and actual response in the frequency domain"""
    def __init__(self,
                 params: np.ndarray,
                 freq_range: tuple,
                 f: float,
                 G_meas: np.ndarray,
                 f_stamp: np.ndarray,
                 G_cov: np.ndarray) -> None:
        """Initialize a instance.

        Attributes:
            U (nr_inputs x N x p x m): the frequency input signals
            Y (nr_outpus x N x p x m): the frequency output signals
            params (nr_outputs x nr_inputs): parameters for the frequency response functions
            freq_range: 
            nr_outputs: the number of outputs channels
            freq_range: the excited frequency range
            f: the sampling frequency
            G_meas(N x nr_outputs x nr_outputs)
            G_cov(N x nr_outputs x nr_inputs): the covariance of the transfer function
        """
        self.params = params
        self.nr_outputs = len(self.params)
        self.nr_inputs = len(self.params[0])
        self.freq_range = freq_range
        self.f = f
        self.G_meas = G_meas
        self.f_stamp = f_stamp
        self.dt = 1/self.f
        self.idx = self.get_freq_index(self.freq_range, 
                                       self.f_stamp)
        self.G_cov = G_cov
    
    @staticmethod
    def get_freq_index(freq_range: tuple,
                       f_stamp: np.ndarray) -> tuple:
        """Get the indices of the start and end
        frequencies in the stamp.
        """        
        return (np.where(f_stamp == freq_range[0])[0][0], 
                np.where(f_stamp == freq_range[1])[0][0])
    
    @staticmethod
    def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
        """Format the axes
        """
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)

    @staticmethod
    def plot_meas(ax: Axes, signal: np.ndarray, stamp: np.ndarray) -> None:
        """Plot one ax.
        """
        ax.scatter(stamp, signal, c='b', marker='x')

    @staticmethod
    def plot_fit(ax: Axes, signal: np.ndarray, stamp: np.ndarray) -> None:
        """Plot one ax.
        """
        ax.plot(stamp, signal, linewidth=1.0, linestyle='-')

    @staticmethod
    def plot_cov(ax: Axes, signal: np.ndarray, stamp: np.ndarray) -> None:
        """Plot the covariance data
        """
        ax.scatter(stamp, signal, c='r', marker='o')
    
    def get_measured_data_per_channel(self, nr_output, nr_input):
        phase = np.angle(self.G_meas[:, nr_output,nr_input], deg=True)
        amp = np.abs(self.G_meas[:, nr_output, nr_input])
        return amp, phase
    
    def get_fit_data_per_channel(self, nr_output, nr_input):
        num = self.params[nr_output][nr_input]['num']
        den = self.params[nr_output][nr_input]['den']
        nr_delay = self.params[nr_output][nr_input]['nr_delay']
        omega = np.linspace(self.freq_range[0], self.freq_range[1], 500)*2*np.pi
        H = signal.TransferFunction(num, den)
        _,  G = H.freqresp(omega)
        s = omega * 1j  
        responce = G * np.exp(-self.dt * s * nr_delay)
        amp = np.abs(responce)
        phase = np.angle(responce, deg=True)
        return amp, phase, omega

    def plot(self) -> None:
        """

        Plot the system identification result and the actual respone of the system.
        Each column corresponds to one output channe. 
        And the first row is about the amplitude and the second row is about phase.

        """
        _, axes = plt.subplots(2*self.nr_inputs, self.nr_outputs, figsize =(4 * self.nr_outputs, 2.5*2*self.nr_inputs))
        for i in range(self.nr_inputs):
            for j in range(self.nr_outputs):
                amp_meas, phase_meas = self.get_measured_data_per_channel(j, i)
                amp_fit, phase_fit, omega = self.get_fit_data_per_channel(j, i)
                cov = self.G_cov[:, j, i]
                # plot the amplitude
                ax = axes[2*i, j]
                ax.set_yscale("log")
                ax.set_xlim(self.freq_range[0]*2*np.pi, 
                            self.freq_range[1]*2*np.pi)
                self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Amplitude')
                self.plot_meas(ax, amp_meas[self.idx[0]+1:self.idx[1]], self.f_stamp[self.idx[0]+1:self.idx[1]]*2*np.pi)
                self.plot_fit(ax, amp_fit, omega)
                self.plot_cov(ax, np.abs(cov[self.idx[0]+1:self.idx[1]]), self.f_stamp[self.idx[0]+1:self.idx[1]]*2*np.pi)
                # plot the phase
                ax = axes[2*i+1, j]
                ax.set_xlim(self.freq_range[0]*2*np.pi, 
                            self.freq_range[1]*2*np.pi)
                self.set_axes_format(ax, r'Radian frequency in $\omega$/$s$', r'Phase')
                self.plot_meas(ax, phase_meas[self.idx[0]+1:self.idx[1]], self.f_stamp[self.idx[0]+1:self.idx[1]]*2*np.pi)
                self.plot_fit(ax, phase_fit, omega)

        plt.tight_layout()
        plt.show()