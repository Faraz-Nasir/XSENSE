from os import environ
import torch

def hardwareModeSafetyCheck(desiredMode):

    if (desiredMode < 0 or desiredMode > 2):
        print(f"Rank: [{GLOBAL_RANK}] :: Faulty hardware more selected.")
        print(f"Rank: [{GLOBAL_RANK}] :: Exiting.")
        exit()

    return int(desiredMode)

#---------------------
# Physical constants
#---------------------
mc2 = 511e3                # electron mass [eV]
c = 2.998e8                # speed of light [m/s]
r = 2.818e-15              # electron radius [m]
q = 1.602e-19              # electron charge [C]
eV2J = 1.602176565e-19     # eV --> Joules
J2eV = 6.242e18            # Joules --> eV
hbar = 6.582e-16           # Planck's constant [eV*s]
hbarJ = 1.0545718e-34      # Planck's constant [J*s]
me = 9.10938356e-31        # electron mass [kg]
alpha = 1.0/137.0          # fine structure constant
eps0 = 8.85e-12            # epsilon 0 (C^2/Nm^2)

CONFIG_FILE_PATH = "XSENSE_Input/config.in"
INPUT_FILE = "XSENSE_Input/ICs_SENSE.in"

DTYPE = torch.float32
# 0 -> GPU ONLY, 1 -> CPU ONLY, 2-> CPU + GPU
HARDWARE_MODE = hardwareModeSafetyCheck(int(environ["HARDWARE_MODE"]))
LOCAL_RANK = int(environ["LOCAL_RANK"])
GLOBAL_RANK = int(environ["RANK"])
LOCAL_WORLD_SIZE = int(environ["LOCAL_WORLD_SIZE"]) # Returns the number of processes running on a single node.--nproc-per-node
WORLD_SIZE = int(environ["WORLD_SIZE"]) # Returns the number for global processes running across the cluster
GROUP_WORLD_SIZE = int(environ["GROUP_WORLD_SIZE"]) # Returns the value for the number of nodes launched i.e --nnodes
MASTER_ADDR = environ["MASTER_ADDR"]
MASTER_PORT = int(environ["MASTER_PORT"])
GROUP_RANK = int(environ["GROUP_RANK"]) # Returns the node-rank 

MASTER_PROCESS_RANK = 0
BACKEND_CPU = 'gloo'
BACKEND_GPU = 'nccl'
CUDA = 'cuda'
CPU = 'cpu'
PARTICLE_COUNT_DTYPE = torch.int64

ANGLES_SENSE_FILE_PATH = f"XSENSE_Output/angles_SENSE:{GROUP_RANK}:{LOCAL_RANK}.txt"

from XSENSE import TotalNpart
if HARDWARE_MODE == 0:
    OUTPUT_FILE_PATH = f"XSENSE_Output/XSENSE:GPU:{LOCAL_WORLD_SIZE}:TotalNpart:{TotalNpart}.txt"
elif HARDWARE_MODE == 1:
    OUTPUT_FILE_PATH = f"XSENSE_Output/XSENSE:CPU:{LOCAL_WORLD_SIZE}:TotalNpart:{TotalNpart}.txt"
elif HARDWARE_MODE == 2:
    OUTPUT_FILE_PATH = f"XSENSE_Output/XSENSE:CPU+GPU:{LOCAL_WORLD_SIZE}:TotalNpart:{TotalNpart}.txt"

OUTPUT_FILE_LINE_FORMAT = r'%15.8e  %15.8e  %15.8e  %15.8e'
