"""
Creation of the gates for quantum circuits 
"""


import numpy as np
import quimb as qu
from quimb.tensor.circuit import rzz_param_gen, rx_gate_param_gen


def H():
    return qu.hadamard()

def RZZ(gamma):
    return rzz_param_gen([gamma])

def RX(beta):
    return rx_gate_param_gen([beta])

def X():
    return qu.pauli('X')

def RZZ(gamma):
    return rzz_param_gen([gamma])

def RZ(beta):
    RZ = np.zeros((2,2,1,2), dtype="complex")
    RZ[0,0,0,0] = 1
    RZ[1,1,0,0] = 1
    RZ[0,0,0,1] = np.cos(-beta*2/2) - 1.0j * np.sin(-beta*2/2)
    RZ[1,1,0,1] = np.cos(-beta*2/2) + 1.0j * np.sin(-beta*2/2)
    return RZ

def CP():
    """COPY gate"""
    CP = np.zeros((2,2,2,1), dtype="complex")
    CP[0,0,0,0] = 1
    CP[1,1,1,0] = 1
    return CP

def ADD():
    """ADD gate"""
    ADD = np.zeros((2,2,2,2), dtype="complex")
    ADD[0,0,0,0] = 1
    ADD[0,0,1,0] = 0
    ADD[0,0,1,1] = 0
    ADD[0,0,0,1] = 1
    ADD[1,1,0,0] = 1
    ADD[1,1,1,0] = 0
    ADD[1,1,0,1] = 0
    ADD[1,1,1,1] = 1
    return ADD