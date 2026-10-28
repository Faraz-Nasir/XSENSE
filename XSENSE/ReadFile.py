from math import atan
from scipy.interpolate import interp1d
from PhysicalConstants import CONFIG_FILE_PATH
#---------------------------------------------------------------------
# Read in parameters of the simulation from the input file config.in.
#---------------------------------------------------------------------
arg_file = open(CONFIG_FILE_PATH, "r")
args = []
for line in arg_file:
    i = 0
    while (line[i:i + 1] != " "):
        i += 1
    num = float(line[0:i])
    args.append(num)

#-------------------
# e-beam parameters
#-------------------
En0 = args[0]
sig_g = args[1]
sigma_e_x = args[2]
sigma_e_y = args[3]
eps_x_n = args[4]
eps_y_n = args[5]

#------------------
# Laser parameters
#------------------
lambda0 = args[6]
sign = args[7]
sigma_p_x = args[8]
sigma_p_y = args[9]
a0 = args[10]
iTypeEnv = int(args[11])
if (iTypeEnv == 3):
    data_file = open("Laser_Envelope_Data.txt", "r")
    exp_data = []
    for line in data_file:
        exp_data.append(line.strip().split())
    exp_xi = []
    exp_a = []
    for line in exp_data:
        exp_xi.append(float(line[0]))
        exp_a.append(float(line[1]))
    exp_f=interp1d(exp_xi,exp_a,kind='cubic')
else:
    exp_xi = []
    exp_f = 0
modType = int(args[12])
fmd_xi=[]
fmd_f=[]
fmdfunc=0
if (modType == 1):
    a0chirp = a0
    fm_param = 0.0
elif (modType == 2):
    a0chirp = a0
    fm_param = 0.0
elif (modType == 3):
    a0chirp = 0.0
    fm_param = float(args[13])
elif (modType == 4):
    a0chirp = 0.0
    fm_param = float(args[13])
elif (modType == 5):
    a0chirp = a0
    fm_param = float(args[13])
elif (modType == 6):
    a0chirp = a0
    fm_param = float(args[13])
elif (modType == 7):
    a0chirp = a0
    fm_param = float(args[13])
elif (modType == 8):
    a0chirp = a0
    fm_param = float(args[13])
elif (modType == 9):
    data_file = open("Fmod_Data.txt", "r")
    fmod_data = []
    for line in data_file:
        fmod_data.append(line.strip().split())
    fmd_xi = []
    fmd_f = []
    for line in fmod_data:
        fmd_xi.append(float(line[0]))
        fmd_f.append(float(line[1]))
    fmdfunc=interp1d(fmd_xi,fmd_f,kind='cubic')
    a0chirp =  float(args[13])
    lambda_RF = 0.0
elif (modType == 10):
    a0chirp = a0
    fm_param = float(args[13])
else:
    a0chirp = 0.0
    fm_param = 0.0
l_angle = args[14]
v_angle = args[15]

pol = args[16]
#---------------------
# Aperture parameters
#---------------------
TypeAp = args[17]
L_aper = args[18]
if (TypeAp == 0):
    R_aper = args[19]
    tmp = args[20]
    theta_max = atan(R_aper/L_aper)
else:
    x_aper = args[19]
    y_aper = args[20]

aper_off_x = args[21]
aper_off_y = args[22]
#-----------------------
# Simulation parameters
#-----------------------
wtilde_min = args[23]
wtilde_max = args[24]
Nout = int(args[25])
Nt = int(args[26])


TotalNpart = int(args[27])

N_MC = int(args[28])
iFile = int(args[29])
n_sub_1 = int(args[30])
n_sub_2 = int(args[31])
iCompton = int(args[32])