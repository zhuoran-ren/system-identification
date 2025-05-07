"""Collection of useful functions.
"""
import os
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation
import numpy as np


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

def quaternion_to_euler(quaternions):
    """
    Convert quaternionds to euler_angles
    Inputs: quaternions: shape (4, N), [x, y, z, w]
    Returns: euler_angles: shape (3, N), [roll, pitch, yaw]
    """

    quats = quaternions.T  # shape: (N, 4), already in [x, y, z, w]
    r = Rotation.from_quat(quats)
    eulers = r.as_euler('xyz', degrees=True)  # Returns roll, pitch, yaw

    return eulers.T  # shape: (3, N)

def convert_data(data, base):
    """
    Args: Original data, shape(7, N)
    Returns: Converted data, shape(6, N)
    """
    positions = data[0:3, :]         # x, y, z
    base_position = base[0:3, :]
    quaternions = data[3:7, :]       # w, x, y, z
    base_quaternions = data[3:7, :]

    # eulers = quaternion_to_euler(quaternions)
    # base = quaternion_to_euler(base_quaternions)
    quat_scipy = quaternions[[0, 1, 2, 3], :]  # 变成 (4, N) 顺序 x, y, z, w

    r0 = Rotation.from_quat(base_quaternions[:, 100])

    # 全部转换为 Rotation 对象
    r_all = Rotation.from_quat(quat_scipy.T)

    # 相对于初始的相对旋转
    r_rel = r_all * r0.inv()

    # 转换为欧拉角（绕 xyz 轴）
    euler = r_rel.as_euler('xyz', degrees=True).T  # shape (3, N)
    return np.vstack((positions - base_position, euler))  # shape: (6, N)

def convert_y_base(base):
    """
    Convert the base signal
    """
    position = base[0:3, :]
    quaternions = base[3:7, :]
    # print(np.mean(quaternions, axis=1))
    # print(np.std(quaternions, axis=1))  # 看看有没有实际变化
    # # eulers = quaternion_to_euler(quaternions)
    # quat_scipy = quaternions[[0, 1, 2, 3], :]  # 变成 (4, N) 顺序 x, y, z, w

    # r0 = Rotation.from_quat(quat_scipy[:, 100])

    # r_all = Rotation.from_quat(quat_scipy.T)

    # r_rel = r_all * r0.inv()

    quat_scipy = quaternions.T            # shape (N, 4) — no reorder needed!

    r = Rotation.from_quat(quat_scipy)
    euler = r.as_euler('xyz', degrees=True).T    # shape (3, N)
    return np.vstack((position, euler)) 


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
    
    return convert_data(data['y_arm1'], data['y_base']),convert_data(data['y_arm2'], data['y_base']), convert_data(data['y_arm3'], data['y_base']), convert_y_base(data['y_base']), data['t_stamp_real']

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
    return data['freq_range'], data['f'], data['N'], data['p'], data['m'], data['u_sysid'], data['us']


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
    y_arm1, y_arm2, y_arm3, y_base, t_stamp = load_response_data(path)
    

    path = os.path.join(root, 'data', 'excitation_signals', file_name)
    freq_range, f, N, p, m, u_sysid, us = load_excitation_data(path)

    return freq_range, f, N, p, m, u_sysid,us,  y_arm1, y_arm2, y_arm3, y_base, t_stamp