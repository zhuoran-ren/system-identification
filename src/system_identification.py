"""Classes for identifying a MIMO system
in the frequency domain.
"""
import numpy as np
import scipy.signal as signal


class SysIdentification():
    """Identify a MIMO system in the frequency domian.
    """
    def __init__(self, freq_range: tuple,
                 f_stamp: np.ndarray,
                 f: float,
                 N: int,
                 p: int,
                 m: int,
                 order_max: int=5,
                 delay_max: int=7) -> None:
        """Initialize a instance.
        
        Args:
            freq_range: the excited frequency range
            f_stamp: the frequency stamp
            f: the sampling frequency
            N: the number of points of each period
            p: the number of repeat times of each signal
            m: the number of different signals
            order_max: the maximum order of the fitted transfer function
            delay_max: the maximum delay of the fitted transfer function

        Attributes:
            dt: the time step
            nr_abandon: the number of abandoned signal when calculating the mean response,
                        considering the transient effect
        """
        self.freq_range = freq_range
        self.f_stamp = f_stamp
        self.f = f
        self.N = N
        self.p = p
        self.m = m
        self.order_max = order_max
        self.delay_max = delay_max

        self.dt = 1/self.f
        self.nr_abandon = 2

    def initialization(self, u: np.ndarray,
                       y: np.ndarray) -> None:
        """Import the inputs and outputs. Initialize and
        calculate the necessary parameters

        Args:
            u (nr_inputs x N*p*m): the inputs applied to the system
            y (nr_outputs x N*p*m): the corresponding outputs
        
        Attributes:
            nr_inputs (int): the number of inputs
            nr_outputs (int): the number of outputs
            U (nr_inputs x N*p*m): the frequency input signals
            Y (nr_outpus x N*p*m): the frequency output signals
            U_split (N x nr_inputs x p x m): recontruct U
            Y_split (N x nr_outputs x p x m): recontruct Y
            U_bar (N x nr_inputs x m): the mean value wrt p times
            Y_bar (N x nr_outputs x m): the mean value wrt p times
        """
        # import the inputs and outputs
        self.u = u
        self.y = y
        # get the number of inputs and outputs
        self.nr_inputs = self.u.shape[0]
        self.nr_outputs = self.y.shape[0]
        # initialize the frequency response matrix
        self.params = [[[{} for _ in range(self.nr_inputs)] for _ in range(self.nr_outputs)]]
        # convert time signal to frequency signal
        self.U = self.time2freq(self.u)
        self.Y = self.time2freq(self.y)
        # convert the signal to (m x nr x p x N)
        self.U_split = self.split_data(self.U, self.N, self.p, self.m)
        self.Y_split = self.split_data(self.Y, self.N, self.p, self.m)
        # calculate the mean value of each repeat
        self.U_bar = self.get_mean_value(self.U_split[:, :, self.nr_abandon:, :])
        self.Y_bar = self.get_mean_value(self.Y_split[:, :, self.nr_abandon:, :])
        # calculate radian frequency: omega = 2\pi*f
        self.omega = None
        # calculate laplace variable: s = j*omega
        self.s = self.omega * 1j  

    @staticmethod
    def split_data() -> np.ndarray:
        """
        """
        pass

    @staticmethod
    def get_mean_value() -> np.ndarray:
        """
        """
        pass

    @staticmethod
    def time2freq(signal: np.ndarray) -> complex:
        """Convert time singal to frequency signal.
        """

    def initialization(self) -> None:
        """Calculate and initialize the necessary parameters.

        Attributes:
            nr_inputs: the number of inputs of the system
            nr_outputs: the number of outputs of the system
        """
        self.nr_inputs = self.u.shape[0]
        self.nr_outputs = self.y.shape[0]
        self.Y = self.time2freq(self.y)
        self.U = self.time2freq(self.U)

    def get_measure_transfer_function(self):
        """
        """
        G_meas = np.zeros((self.N, self.nr_outputs, self.nr_inputs))
        for i in range(self.N):
            G_meas[i, :, :] = None
        return G_meas

    @staticmethod
    def compensate_delay(G: complex,
                         nr_delay: int,
                         dt: float,
                         s: complex) -> complex:
        """Compensate the effect of the delay.

        Args:
            G: the measured points of the transfer function with delays
            nr_delay: the number of delays
            dt: the time step
            s: the laplace variable
        
        Returns:
            G_comp: the measured points of the transfer function without delays
        """
        return G * np.exp(nr_delay*dt*s)

    @staticmethod
    def fit_transfer_function(G: complex,
                              order_num: int,
                              order_den: int,
                              s: complex) -> tuple[np.ndarray,
                                                   np.ndarray]:
        """Fit a transfer function to G using specified model.

        Args:
            G: the measured points of the system
            order_num: the order of the numerator
            order_den: the order of the denominator
            s: the Laplace variable
        
        Returns:
            num: the parameters of the numerator
            den: the parameters of the denominator
        """
        # calculate the constant column
        const_vec = -G * (s ** order_den)
        # construct the V matrix
        V = np.column_stack([s**i for i in range(order_num, -1, -1)]) 
        V = np.column_stack([V] + [-G * (s**i) for i in range(order_den - 1, -1, -1)])
        # split into real part and imaginary part
        const_vec = np.concatenate([const_vec.real, const_vec.imag])
        V = np.vstack([V.real, V.imag])
        # fit the parameters
        param, _, _, _ = np.linalg.lstsq(V, -const_vec, rcond=None)
        # extract the parameters
        num = param[:order_num+1].flatten()
        den = np.concatenate(([1], param[order_num+1:].flatten())) 
        return num, den

    @staticmethod
    def get_transfer_function(num: np.ndarray,
                              den: np.ndarray,
                              nr_delay: int,
                              dt: float,
                              s: complex) -> complex:
        """Return the transfer function with delays.

        Args:
            num: the parameters of the numerator
            den: the parameters of the denominator
            nr_delay: the number of delays
            dt: the time step
            s: the Laplace variable
        
        Returns:
            G: the transfer function with delays
        """
        return signal.TransferFunction(num, den) * np.exp(-dt * s * nr_delay)
    
    @staticmethod
    def get_amp_difference(amp1: np.ndarray,
                           amp2: np.ndarray) -> float:
        """Return the amplitude difference. In dB.
        """
        return 20 * np.abs(np.log10(amp2) - np.log10(amp1))

    @staticmethod
    def get_phase_difference(phase1: np.ndarray,
                             phase2: np.ndarray) -> float:
        """Return the phase difference. In degree.
        """
        return np.abs(phase2 - phase1)

    def get_loss(self, G1: complex, 
                 G2: complex,
                 omega: complex,
                 eta_amp: float=0.5,
                 eta_phase: float=0.5) -> float:
        """Compare the different between two transfer
        functions. Will comprehensively consider the
        difference of amplitude and phase.

        Args:
            G1: the first transfer function
            G2: the second transfer function
            omega: the radian frequency
            eta_amp: the weight of the amplitude difference
            eta_phase: the weight of the phase difference
        
        Returns:
            loss: the difference between two transfer functions
        """
        # get the amplitude and phase of G1
        _, amp1, phase1 = signal.bode(G1, omega)
        # get the amplitude and phase of G2
        _, amp2, phase2 = signal.bode(G2, omega)
        # calculate the amplitude difference
        amp_diff = self.get_amp_difference(amp1, amp2)
        # calculate the phase differnce
        phase_diff = self.get_phase_difference(phase1, phase2)
        # calculate RMS error
        return eta_amp*np.sqrt(np.mean(amp_diff ** 2) + eta_phase*np.mean(phase_diff ** 2))

    def _identify_channel(self, G: complex, 
                          nr_delay: int,
                          order_num: int,
                          order_den: int) -> tuple[float,
                                                   np.ndarray,
                                                   np.ndarray]:
        
        G_comp = self.compensate_delay(G, nr_delay, self.dt, self.s)
        num, den = self.fit_transfer_function(G_comp, order_num, order_den, self.s)
        G_est = self.get_transfer_function(num, den, nr_delay)
        loss = self.get_loss(G, G_est, self.omega)
        return loss, num, den
    
    def identify_channel(self, G: complex) -> dict:
        """
        """
        data = [[[{'loss': 10000} for _ in range(self.delay_max)] for _ in range(self.order_max)] for _ in range(self.order_max)]
        
        for nr_delay in range(self.delay_max):
            for order_den in range(1, self.order_max):
                for order_num in range(order_den+1):
                    loss, num, den = self._identify_channel(G=G,
                                                            nr_delay=nr_delay,
                                                            order_num=order_num,
                                                            order_den=order_den)

                    data[nr_delay][order_den][order_num] = {
                        'loss': loss,
                        'order_num': order_num,
                        'order_den': order_den,
                        'nr_delay': nr_delay,
                        'num': num,
                        'den': den,
                    }
                    
    def identify_system(self) -> None:
        """Identify the system based on the given
        inputs and outputs in the frequency domian.

        Args:
        """
        # calculate the average response signal
        self.G_meas = self.get_measure_transfer_function()
        self.G_cov = None
        for row in range(self.nr_outputs):
            for col in range(self.nr_inputs):
                self.params[row][col] = self.identify_channel(self.G_meas[:, row, col])