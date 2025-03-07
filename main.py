"""This script is used to identify a system
in the frequency domain.
"""
from src.system_identification import SysIdentification
from src.utils import *

def main(file_name):
    # load the file, including input and output signals
    # and the parameters for the excitation signals
    freq_range, f, N, p, m, u, y = load_identification_data(file_name)
    sysid = SysIdentification(freq_range=freq_range,
                                           f=f,
                                           N=N,
                                           p=p,
                                           m=m)
    sysid.import_data(u=u, y=y)
    sysid.initialization()
    sysid.identify_system()
    
if __name__ == '__main__':
    main()