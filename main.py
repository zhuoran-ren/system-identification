"""This script is used to identify a system
in the frequency domain.
"""
from src.system_identification import SysIdentification
from src.utils import *
from src.visualization import Visualization


def main(file_name):
    # load the file, including input and output signals
    # and the parameters for the excitation signals
    freq_range, f, N, p, m, u, y = load_identification_data(file_name)
    sysid = SysIdentification(freq_range=freq_range,
                              f=f, N=N, p=p, m=m)
    sysid.initialization(u=u, y=y)
    sysid.identify_system()
    
    print('here')
    # TODO: compare to the true value 
    visualization = Visualization(f=f, 
                                  G_meas= sysid.G_meas,
                                  params=sysid.params,freq_range=freq_range, 
                                  f_stamp = sysid.f_stamp, G_cov = sysid.G_cov
                                  )
    visualization.plot()


if __name__ == '__main__':
    main(file_name='test')