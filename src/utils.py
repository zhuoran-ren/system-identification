"""Collection of useful functions.
"""
import os
from pathlib import Path
import pickle

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it
    if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def load_response_data(path: Path) -> tuple:
    """Load the response data from file.

    Args:
        path: path to the file
    
    Returns:
        system: the name of the system
        signal: the name of the excited signal
        u: the inputs
        y: the corresponding outputs
        t_stamp_input: the time stamp of the inputs
        t_stamp_output: the time stamp of the outputs
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['system'], data['signal'], data['u'], data['y'], data['t_stamp_input'], data['t_stamp_output']

def load_excitation_data(path: Path) -> tuple:
    """Load the parameters of excitation signals from file.

    Args:
        path: path to the file

    Returns:
        freq_range: the excited frequency range
        f: the sampling frequency
        N: the number of points of each signal
        p: the number of repeat times
        m: the number of different signals
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['freq_range'], data['f'], data['N'], data['p'], data['m']

def load_identification_data(file_name: str) -> tuple:
    """Load data for identification.

    Args:
        file_name: name of the identification experiment

    Returns:
        freq_range: the excited frequency range
        f: the sampling frequency
        N: the number of points of each signal
        p: the number of repeat times
        m: the number of different signals
        u: the inputs applied to the system
        y: the corresponding outputs
    """
    root = get_parent_path(lvl=1)
    path = os.path.join(root, 'data', 'response_signals', file_name)
    _, signal_name, u, y, _, _ = load_response_data(path)

    path = os.path.join(root, 'data', 'excitation_signals', signal_name)
    freq_range, f, N, p, m = load_excitation_data(path)

    return freq_range, f, N, p, m, u, y