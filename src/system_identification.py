"""Classes for identifying a MIMO system
in the frequency domain.
"""
import numpy as np
import scipy.signal as signal

class SysIdentification():
    """Identify a MIMO system in the frequency domian.
    """
    def __init__(self, freq_range: tuple,
                 f: float,
                 N: int,
                 p: int,
                 m: int,
                 order_max: int=5,
                 delay_max: int=7) -> None:
        """Initialize an instance.
        
        Args:
            freq_range: the excited frequency range
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
        self.f = f
        self.N = N
        self.p = p
        self.m = m
        self.order_max = order_max
        self.delay_max = delay_max

        self.dt = 1/self.f
        self.nr_abandon = 2

    @staticmethod
    def get_freq_stamp(f: float, 
                       N: int) -> np.ndarray:
        """Calculate the frequency stamp: f_i = i*(f/N).
        """
        return np.arange(0, N) / N * f
    
    @staticmethod
    def get_freq_index(freq_range: tuple,
                       f_stamp: np.ndarray) -> tuple:
        """Get the indices of the start and end
        frequencies in the stamp.
        """        
        return (np.where(f_stamp == freq_range[0])[0][0], 
                np.where(f_stamp == freq_range[1])[0][0])
    
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
            f_stamp: the frequency stamp
            idx: the start and end indices in the frequency stamp
            params (nr_outputs x nr_inputs): parameters for the frequency response functions
            U_split (N x nr_inputs x p x m): recontruct U
            Y_split (N x nr_outputs x p x m): recontruct Y
            U (nr_inputs x N x p x m): the frequency input signals
            Y (nr_outpus x N x p x m): the frequency output signals
            U_bar (N x nr_inputs x m): the mean value wrt (p-nr_abandon) times
            Y_bar (N x nr_outputs x m): the mean value wrt (p-nr_abandon) times
            omega: the valid radian frequency stamp based on the frequency range
            s: the Laplace variable: s = j * omega
            G_meas (N x nr_outputs x nr_inputs): the measured response at each frequency
            G_cov (nr_outputs*nr_inputs x nr_outputs*nr_inputs): the covariance matrix of the transfer functions         
        """
        # import the inputs and outputs
        self.u = u
        self.y = y
        # get the number of inputs and outputs
        self.nr_inputs = self.u.shape[0]
        self.nr_outputs = self.y.shape[0]
        # get the frequency stamp
        self.f_stamp = self.get_freq_stamp(self.f, self.N)
        # get the start and end indices in the frequency stamp
        self.idx = self.get_freq_index(self.freq_range, self.f_stamp)
        # initialize the frequency response matrix
        self.params = [[{} for _ in range(self.nr_inputs)] for _ in range(self.nr_outputs)]
        # convert the signal to (m x nr x p x N)
        self.u_split = self.split_data(self.u, self.N, self.p, self.m)
        self.y_split = self.split_data(self.y, self.N, self.p, self.m)
        # convert time signal to frequency signal
        self.U = self.time2freq(self.u_split, axis=-1)
        self.Y = self.time2freq(self.y_split, axis=-1)
        # calculate the mean value of each repeat
        self.U_bar = np.transpose(self.get_mean_value(self.U[:, :, self.nr_abandon:, :], axis=2), (2, 0, 1))
        self.Y_bar = np.transpose(self.get_mean_value(self.Y[:, :, self.nr_abandon:, :], axis=2), (2, 0, 1))
        # calculate radian frequency: omega = 2\pi*f
        self.omega = self.f_stamp[self.idx[0]:self.idx[1]+1] * 2 * np.pi
        # calculate laplace variable: s = j*omega
        self.s = self.omega * 1j  
        # calculate the average response signal
        self.G_meas = self.get_measure_transfer_function(self.U_bar, self.Y_bar)
        # compensate the DC component
        self.G_meas[0, :, :] = self.G_meas[1, :, :]
        # calculate the covariance of G
        self.G_cov = self.get_covariance()

    
    def compute_covariance_y(self,
                             id_n: int) -> np.ndarray:
        """Caculate the covariance of the outputs

        Args:
            p: The total number of repetitions for each experiments
            U (nr_inputs x m x p x N): the frequency input signals
            Y (nr_outpus x m x p x N): the frequency output signals
            U_bar (N x nr_inputs x m): the mean value wrt (p-nr_abandon) times
            Y_bar (N x nr_outputs x m): the mean value wrt (p-nr_abandon) times

        Returns:
            Cy (nr_outputs * nr_outputs): Covariance matrix of the output data
        """
        
        Cy = np.zeros((self.nr_outputs, self.nr_outputs), dtype=np.complex64)
        Y_mean_over_m = np.mean(self.Y, axis=1)
        Y_bar = np.mean(Y_mean_over_m, axis=1)
        for i in range(self.nr_outputs):
            for j in range(self.nr_outputs):
                Cy[i, j] = (Y_mean_over_m[i, :, id_n] - Y_bar[i, id_n]).T @ np.conjugate((Y_mean_over_m[j, :, id_n] - Y_bar[i, id_n])) / (self.p - 1)
        return Cy
    
    def compute_covariance_u(self,
                             id_n: int) -> np.ndarray:
        """Caculate the covariance of the inputs

        Returns:
            Cu (nr_inputs * nr_inputs): Covariance matrix of the input data

        """
        Cu = np.zeros((self.nr_inputs, self.nr_inputs), dtype=np.complex64)
        U_mean_over_m = np.mean(self.U, axis=1)
        U_bar = np.mean(U_mean_over_m, axis=1)
        for i in range(self.nr_inputs):
            for j in range(self.nr_inputs):
                Cu[i, j] = (U_mean_over_m[i, :, id_n] - U_bar[i, id_n]).T @ np.conjugate(U_mean_over_m[j, :, id_n] - U_bar[j, id_n]) /(self.p -1)
        return Cu
    
    def compute_covariance_yu(self,
                              id_n: int) -> np.ndarray:
        """Caculate the covariance of the inputs and outputs

        Returns:
            Cyu (nr_outputs * nr_inputs): Cross-covariance matrix between output and input data

        """
        Cyu = np.zeros((self.nr_outputs, self.nr_inputs), dtype=np.complex64)

        Y_mean_over_m = np.mean(self.Y, axis=1)
        Y_bar = np.mean(Y_mean_over_m, axis=1)
        U_mean_over_m = np.mean(self.U, axis=1)
        U_bar = np.mean(U_mean_over_m, axis=1)

        for i in range(self.nr_outputs):
            for j in range(self.nr_inputs):
                Cyu[i, j] = (Y_mean_over_m[i, :, id_n] - Y_bar[i, id_n]).T @ np.conjugate(U_mean_over_m[j, :, id_n]- U_bar[j,id_n]) / (self.p - 1)
        return Cyu
    
    def compute_covariance(self,
                           id_n: int) -> np.ndarray:
        Cu = self.compute_covariance_u(id_n)
        Cy = self.compute_covariance_y(id_n)
        Cyu = self.compute_covariance_yu(id_n)
        upper_row = np.hstack([Cy, Cyu])
        lower_row = np.hstack([Cyu.T, Cu]) 
        C = np.vstack([upper_row, lower_row])
        return C
    
    def get_covariance_n(self,
                       id_n: int) -> np.ndarray:
        """Calculate the covariance of the transfer functions.

        Args:
            self.U(nr_outputs x m x p x N)
            
        Returns:
            G_cov(nr_outputs * nr_inputs * nr_outputs * nr_inputs): the covariacne matrix of the transfer functions
        """
        I_ny = np.eye(self.nr_outputs)
        Vk = np.hstack([I_ny, -self.G_meas[id_n, :, :]])
        Ck = self.compute_covariance(id_n)
        element_2 = Vk @ Ck @ (Vk.conj().T)
   
        element_1 = np.linalg.inv((self.U_bar[id_n, :, :] @ (self.U_bar[id_n, :, :].conj().T)).conj())
        cov = np.kron(element_1, element_2)
        return cov

    def get_covariance(self) -> np.ndarray:
        covariance =np.zeros((self.N, self.nr_outputs, self.nr_inputs), dtype=np.complex64)
        for id_n in range(self.idx[0], self.idx[1] + 1):
            cov = self.get_covariance_n(id_n)
            k = 0
            for i in range(self.nr_inputs):
                for j in range(self.nr_outputs):
                    covariance[id_n, j, i] = cov[k, k]
                    k = k + 1
        return covariance     


    @staticmethod
    def split_data(a: np.ndarray,
                   N: int,
                   p: int,
                   m: int) -> np.ndarray:
        """Convert a nr_channels x nr_points signal into
        a nr_channels x m x p x N signal, where 
        nr_points = N*m*p.

        Args:
            a (nr_channels x nr_points): the given signal
            N: the number of points
            p: the repeat times
            m: the number of different signals
        
        Returns:
            b (nr_channels x m x p x N): the re-constructed signal
        """
        nr_channels, nr_points = a.shape
        return a.reshape(nr_channels, m, p, N)

    @staticmethod
    def get_mean_value(a: np.ndarray,
                       axis: int) -> np.ndarray:
        """Calculate the average value along the desired
        axis.

        Args:
            a: the given data
            axies: the desired axis

        Returns:
            b: the averaged data
        """
        return np.mean(a, axis=axis)

    @staticmethod
    def time2freq(a: np.ndarray,
                  axis: int) -> complex:
        """Convert time singal to frequency signal
        along the given axis.

        Args:
            a: the given time signal
            axis: the desired axis
        
        Returns:
            b: the frequency signal
        """
        return np.fft.fft(a, axis=axis)

    def get_measure_transfer_function(self, 
                                      U: complex,
                                      Y: complex) -> complex:
        """Calculate the measured response based on
        multiple experiments.

        Args:
            U (N x nr_inputs x m): the averaged input signals in the frequency domain
            Y (N x nr_outpus x m): the averaged output signals in the frequency domain
        
        Returns:
            G (N x nr_outpus x nr_inputs): the measured response at each frequency
        """
        N, nr_inputs, m = U.shape
        U_pinv = np.linalg.pinv(U.reshape(-1, nr_inputs, m)).reshape(N, m, nr_inputs) 
        return np.einsum('nij,njk->nik', Y, U_pinv) 

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
                              den: np.ndarray) -> complex:
        """Return the transfer function with delays.

        Args:
            num: the parameters of the numerator
            den: the parameters of the denominator
        
        Returns:
            G: the transfer function with delays
        """
        return signal.TransferFunction(num, den)
    
    @staticmethod
    def get_amp_difference(amp1: np.ndarray,
                           amp2: np.ndarray) -> float:
        """Return the amplitude difference. In dB.
        """
        return 20 * np.abs(np.log10(amp2) - np.log10(amp1))

    @staticmethod
    def get_phase_difference(phase1: np.ndarray,
                             phase2: np.ndarray,
                             mode: str) -> float:
        """Return the phase difference. In degree.
        """
        if mode == 'degree':
            return np.abs(phase2 - phase1)
        elif mode == 'radian':
            return np.abs((phase2 - phase1)*180/np.pi)
        
    @staticmethod
    def get_amp(G: np.ndarray) -> np.ndarray:
        """Return the amplitude of the measured points.
        """
        return np.abs(G)
    
    @staticmethod
    def get_phase(G: np.ndarray,
                  mode: str='degree') -> np.ndarray:
        """Return the phase of the measured points in
        radian or degree.
        """
        if mode == 'degree':
            return np.angle(G, deg=True)
        elif mode == 'radian':
            return np.angle(G)

    def get_amp_phase(self, G: np.ndarray,
                      mode: str) -> tuple[np.ndarray,
                                          np.ndarray]:
        """Calculate the amplitude and phase based
        on the given measured points.

        Args:
            G: the measured points
            mode: degree or radian
        
        Returns:
            amp: the amplitude in the frequency domain
            phase: the phase in the frequency domain
        """
        return self.get_amp(G), self.get_phase(G, mode)

    def get_loss(self, G1: complex, 
                 G2: complex,
                 mode: str='degree',
                 eta_amp: float=0.5,
                 eta_phase: float=0.5) -> float:
        """Compare the different between two transfer
        functions. Will comprehensively consider the
        difference of amplitude and phase.

        Args:
            G1: the measured points of the first transfer function
            G2: the measured points of the second transfer function
            eta_amp: the weight of the amplitude difference
            eta_phase: the weight of the phase difference
        
        Returns:
            loss: the difference between two transfer functions
        """
        # get the amplitude and phase of G1
        amp1, phase1 = self.get_amp_phase(G1, mode)
        # get the amplitude and phase of G2
        amp2, phase2 = self.get_amp_phase(G2, mode)
        # calculate the amplitude difference
        amp_diff = self.get_amp_difference(amp1, amp2)
        # calculate the phase differnce
        phase_diff = self.get_phase_difference(phase1, phase2, mode)
        # calculate RMS error
        return eta_amp*np.sqrt(np.mean(amp_diff ** 2) + eta_phase*np.mean(phase_diff ** 2))

    # def get_loss(self, H1: complex, 
    #              H2: complex,
    #              omega: complex,
    #              eta_amp: float=0.5,
    #              eta_phase: float=0.5) -> float:
    #     """Compare the different between two transfer
    #     functions. Will comprehensively consider the
    #     difference of amplitude and phase.

    #     Args:
    #         G1: the first transfer function
    #         G2: the second transfer function
    #         omega: the radian frequency
    #         eta_amp: the weight of the amplitude difference
    #         eta_phase: the weight of the phase difference
        
    #     Returns:
    #         loss: the difference between two transfer functions
    #     """
    #     # get the amplitude and phase of H1
    #     _, amp1, phase1 = signal.bode(H1, omega)
    #     # get the amplitude and phase of H2
    #     _, amp2, phase2 = signal.bode(H2, omega)
    #     # calculate the amplitude difference
    #     amp_diff = self.get_amp_difference(amp1, amp2)
    #     # calculate the phase differnce
    #     phase_diff = self.get_phase_difference(phase1, phase2)
    #     # calculate RMS error
    #     return eta_amp*np.sqrt(np.mean(amp_diff ** 2) + eta_phase*np.mean(phase_diff ** 2))
    
    def get_frequency_response_without_delay(self, 
                                             num: np.ndarray,
                                             den: np.ndarray) -> complex:
        """Build the transfer function based on num and den.
        Then calculate the response at certain frequency without
        delay.

        Args: 
            num: the parameters of the numerator
            den: the parameters of the denominator

        Returns:
            G: the response at certrain frequency without delay effect
        """
        H = self.get_transfer_function(num, den)
        _,  G = H.freqresp(self.omega)
        return G


    def get_frequency_response_with_delay(self, num: np.ndarray,
                                                den: np.ndarray,
                                                nr_delay: int,
                                                dt: float,
                                                s: complex) -> complex:
        """Calculate the response at certain frequency,
        considering the effect of delay.

        Args: 
            num: the parameters of the numerator
            den: the parameters of the denominator
            nr_delay: the number of delays
            dt: the time step
            s: the Laplace variable

        Returns:
            G: the response at certrain frequency considering delay effect
        """
        return self.get_frequency_response_without_delay(num, den) * np.exp(-dt * s * nr_delay)

    def fit_channel(self, G: complex, 
                          nr_delay: int,
                          order_num: int,
                          order_den: int) -> tuple[float,
                                                   np.ndarray,
                                                   np.ndarray]:
        # compensate the effect of delay
        G_comp = self.compensate_delay(G, nr_delay, self.dt, self.s)
        # fit the transfer function, get corresponding parameters
        num, den = self.fit_transfer_function(G_comp, order_num, order_den, self.s)
        # get the fitted transfer function considering the effect of delay
        G_est = self.get_frequency_response_with_delay(num, den, nr_delay, self.dt, self.s)
        # calculate the loss
        loss = self.get_loss(G, G_est)
        return loss, num, den
    
    @staticmethod
    def select_min_index(a: list,
                         key: str) -> tuple:
        """Return the indices of the minimum value
        of the given key in the list.

        Args:
            a: the given list
            key: the desired key
        
        Returns:
            idx: the index
        """
        loss_array = np.array([[[d.get(key, float("inf")) for d in row] for row in layer] for layer in a])
        min_index = np.unravel_index(np.argmin(loss_array), loss_array.shape)
        return min_index

    def identify_channel(self, G: complex) -> dict:
        """Fit a transfer function for one entry of the 
        frequency response matrix (FRM).

        Args:
            G: the measured points of one entry of FRM, only consider the valid range
        
        Returns:
            data: the parameters of the fitted transfer function
        """
        # data <- [delay][den][num]
        data = [[[{'loss': 10000} for _ in range(self.order_max)] for _ in range(self.order_max)] for _ in range(self.delay_max)]
        
        for nr_delay in range(self.delay_max):
            for order_den in range(1, self.order_max):
                for order_num in range(order_den+1):
                    loss, num, den = self.fit_channel(G=G,
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
        
        (a1, a2, a3) = self.select_min_index(data, 'loss')
        return data[a1][a2][a3]
                    
    def identify_system(self) -> None:
        """Identify the system based on the given
        inputs and outputs in the frequency domian.

        Args:
        """
        for row in range(self.nr_outputs):
            for col in range(self.nr_inputs):
                self.params[row][col] = self.identify_channel(self.G_meas[self.idx[0]:self.idx[1]+1, row, col])