#==============================================================================
# XSENSE (eXtended Simulation of Emitted Nonlinear Scattering Events) code
#
# PyTorch GPU-accelerated version
#
# Computes inverse Compton scattering spectrum in the non-linear regime.
# Described in detail in
# - Terzic et al. 2019, EPL 126, 12003
# - Terzic et al. 2021
# Based on work reported in:
# - Krafft 2004, PRL 92, 204802
# - Terzic, Deitrick, Hofler & Krafft 2016, PRL 112, 074801
#
# INPUT:  config.in (parameters for e-beam, laser, aperture and simulation)
# OUTPUT: output_SENSE.txt
#   Column 1: E' [eV]
#   Column 2: omega/omega'
#   Column 3: dN/dE' [1/eV] per electron in the e-beam
#             (to get the total number of scattered photons, multiply this
#              column by the total number of electrons in the beam: Q/q,
#              where Q is total charge and q the charge of an electron)
#   Column 4: dN/dE' normalized to 1
#
# PyTorch GPU version with optimized memory transfers
#==============================================================================
from PhysicalConstants import *
from RankMechanism.RankCommunication import RankCommunication
from BatchingMechanism.Batching import BatchingInformation
import timeit
import math 
import torch
from sys import exit
from torch import Generator
from math import pi,cos,sqrt,exp,fabs
from ReadFile import *
import numpy as np
import scipy.integrate as integrate
from RankMechanism.RankInformation import RankInformation
from RankMechanism.RanksInformationDistribution import RanksInformationDistribution
from Prototype.Atomics import Atomics
#---------------------
# Global Properties
#---------------------

rankCommunication = RankCommunication(LOCAL_RANK, GLOBAL_RANK, HARDWARE_MODE)
DEVICE = rankCommunication.getDevice()
GENERATOR = Generator(device=DEVICE)

def setSeedValueForGenerator():
    GENERATOR.manual_seed(1000+GLOBAL_RANK)

np.set_printoptions(threshold=np.inf)

def _compute_ntot_batch_impl(ax_scaled, axi_scaled, axi2_scaled, xi_batch,
                              omLam_batch, c0, c2, c3, c4):
    """
    Inner kernel for Ntot batch computation.
    Fuses operations to reduce memory traffic.
    """
    arg = (c0[:, :, :, None] * omLam_batch[:, :, None, None] *
           (xi_batch[None, None, None, :] * c2[:, :, :, None] -
            axi_scaled[:, None, None, :] * c3[:, :, :, None] +
            axi2_scaled[:, None, None, :] * c4[:, :, :, None]))

    ca = torch.cos(arg)
    sa = torch.sin(arg)

    ax_exp = ax_scaled[:, None, None, :]
    ax2_exp = ax_exp ** 2

    n1x = (ax_exp * ca).sum(dim=3)
    n2x = (ax_exp * sa).sum(dim=3)
    n1z = (ax2_exp * ca).sum(dim=3)
    n2z = (ax2_exp * sa).sum(dim=3)

    return n1x, n2x, n1z, n2z

def _compute_final_values_impl(n1x, n2x, n1z, n2z, c1_exp, c2, c0, f_val_exp, ct_exp,
                                betaz_exp, sp_exp, cp_exp, st_exp):
    """
    Kernel for final value computation after Ntot accumulation.
    """
    Dx = torch.sqrt(n1x**2 + n2x**2) / c1_exp
    Dz = torch.sqrt(n1z**2 + n2z**2) / (2.0 * (c1_exp**2) * c2)

    bracket = 1.0 + f_val_exp * (1.0 + ct_exp) * c2 * c0 / (1.0 - betaz_exp * ct_exp)

    term1 = 0.5 * (1.0/c0 + c0) - 1.0
    term2 = ((ct_exp - betaz_exp) / (1.0 - betaz_exp * ct_exp)) * cp_exp

    D_perp = (Dx**2) * (term1 + 2 * (sp_exp**2))
    D_par = (Dx**2) * term1 + 2 * ((Dx * term2 + Dz * st_exp)**2)

    return (D_perp + D_par) * bracket


def determineCompileMode():
    if DEVICE.type == CUDA:
        return "max-autotune"
    else:
        return "reduce-overhead"
    
TORCH_COMPILE_MODE = determineCompileMode()

USE_TORCH_COMPILE = True

# Create compiled versions
if USE_TORCH_COMPILE:
    _compute_ntot_batch = torch.compile(_compute_ntot_batch_impl, mode=TORCH_COMPILE_MODE)
    _compute_final_values = torch.compile(_compute_final_values_impl, mode=TORCH_COMPILE_MODE)
else:
    _compute_ntot_batch = _compute_ntot_batch_impl
    _compute_final_values = _compute_final_values_impl

#-------------------------------
# Laser beam envelope function
#-------------------------------
def alpha_x(aPeak, xi, sign, iTypeEnv, exp_f, exp_xi):
    """Laser beam envelope function - fully vectorized."""
    xis2 = (xi ** 2) / (2.0 * (sign ** 2))
    gaussian_val = aPeak * torch.exp(-xis2)

    if iTypeEnv == 1:
        return gaussian_val
    elif iTypeEnv == 2:
        mask = (xi > 0.0) & (xi < sign)
        return torch.where(mask, torch.full_like(xi, aPeak), torch.zeros_like(xi))
    elif iTypeEnv == 3:
        mask = (xi < exp_xi[0]) | (xi > exp_xi[-1])
        return torch.where(mask, torch.zeros_like(xi), exp_f(xi))
    else:
        return gaussian_val

#-------------------------------------------
# Efficient evaluation of an error function
#-------------------------------------------
def erf_approx(x):
    """
    Efficient approximation of the error function.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign_val = 1
    if x < 0:
        sign_val = -1
    x = fabs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * exp(-x*x)
    return sign_val * y

#-------------------------------
# Frequency modulation function
#-------------------------------
def A_func(Y, a0):
    return (a0**2) * exp(-2.0 * (Y**2)) / 2.0

def s_1(Y, a0, p):
    return (p * A_func(Y, a0) / 2.0 + (p**3.0) / 27.0
           + sqrt((p**4) * A_func(Y, a0) / 27.0 + (p**2) * (A_func(Y, a0)**2) / 4.0))**(1.0/3.0)

def s_2(Y, a0, p):
    return (p * A_func(Y, a0) / 2.0 + (p**3.0) / 27.0
           - sqrt((p**4) * A_func(Y, a0) / 27.0 + (p**2) * (A_func(Y, a0)**2) / 4.0))**(1.0/3.0)

def fmod(xi, modType, sign, a0chirp, fm_param, beta, fmdfunc, fmd_xi, fmd_f):
    """Frequency modulation - fully vectorized with torch.where for branch-free conditionals."""
    eps = 1e-10
    safe_xi = torch.where(xi.abs() < eps, torch.ones_like(xi), xi)  # avoid div/0

    if modType == 0:
        return torch.ones_like(xi)

    elif modType == 1:
        const1 = 1.0 / (1.0 + 0.5 * a0chirp**2)
        val = const1 * (1.0 + a0chirp**2 * pi**0.5 * sign / (4.0 * safe_xi) * torch.erf(safe_xi / sign))
        return torch.where(xi.abs() < eps, torch.ones_like(xi), val)

    elif modType == 2:
        val = 1.0 + a0chirp**2 * pi**0.5 * sign / (4.0 * safe_xi) * torch.erf(safe_xi / sign)
        return torch.where(xi.abs() < eps, torch.ones_like(xi), val)

    elif modType == 3:
        lambda_RF = fm_param
        return 1.0 - 4.0 * pi**2 * xi**2 / (3.0 * lambda_RF**2)

    elif modType == 4:
        lambda_RF = fm_param
        eps2 = 1e-5
        safe_xi2 = torch.where(xi.abs() > eps2, xi, torch.ones_like(xi))
        val = 0.6666667 * (1.0 + (lambda_RF / (8.0 * pi * safe_xi2)) * torch.sin(4.0 * pi * safe_xi2 / lambda_RF))
        return torch.where(xi.abs() > eps2, val, torch.ones_like(xi))

    elif modType == 5:
        p = fm_param
        eps2 = 1e-5
        xi_np = memoryTransferFromGPUToCPU(DEVICE,xi)
        tmp0 = integrate.quad(lambda x: s_1(x, a0chirp, p) + s_2(x, a0chirp, p), 0, eps2)[0]
        def _fmod_scalar(xi_val):
            Y = xi_val / (2.0**0.5 * sign)
            tmp = integrate.quad(lambda x: s_1(x, a0chirp, p) + s_2(x, a0chirp, p), 0, Y)[0]
            return (p/3.0 + tmp/Y) / (p/3.0 + tmp0/eps2) if abs(Y) > eps2 else 1.0
        result = np.vectorize(_fmod_scalar)(xi_np)
        
        return torch.tensor(result, dtype=DTYPE, device=DEVICE)

    elif modType == 6:
        theta_FM = fm_param
        ct = cos(theta_FM)
        f0 = 1.0 / (1.0 + 0.5 * a0chirp**2)
        const1 = (1.0 - beta * ct) / (1.0 - beta)
        const2 = ((1.0 + ct) / 2.0) * a0chirp**2 * pi**0.5 * sign
        val = (f0 / const1) * (const1 + const2 * torch.erf(safe_xi / sign) / (4.0 * safe_xi))
        return torch.where(xi.abs() < eps, torch.ones_like(xi), val)

    elif modType == 7:
        theta_FM = fm_param
        ct = cos(theta_FM)
        const1 = (1.0 - beta * ct) / (1.0 - beta)
        const2 = ((1.0 + ct) / 2.0) * a0chirp**2 * pi**0.5 * sign
        at_zero = 1.0 + (a0chirp**2 / 2.0) * (1.0 - beta) * (1.0 + ct) / (2.0 * (1.0 - beta * ct))
        val = (1.0 / const1) * (const1 + const2 * torch.erf(safe_xi / sign) / (4.0 * safe_xi))
        return torch.where(xi.abs() < eps, torch.full_like(xi, at_zero), val)

    elif modType == 8:
        return torch.clamp(1.0 - fm_param * xi.abs(), min=0.0)

    elif modType == 9:
        xi_np = memoryTransferFromGPUToCPU(DEVICE,xi)
        out = np.where((xi_np < fmd_xi[0]) | (xi_np > fmd_xi[-1]), fmd_f[0], fmdfunc(xi_np))
        
        return torch.tensor(out, dtype=DTYPE, device=DEVICE)

    elif modType == 10:
        val = (2.0 / 3.0) * (pi**0.5 * sign / (4.0 * safe_xi)) * torch.erf(safe_xi / sign)
        return torch.where(xi.abs() < eps, torch.ones_like(xi), val)

    else:
        return torch.ones_like(xi)

#---------------------------------------------------------------
# Tabulate inner A(xi) integrals (Terzic et al. 2019: Eq. (3)).
#---------------------------------------------------------------
def tabulate1_ax(iTypeEnv, exp_f, exp_xi, modType, fmdfunc, fmd_xi, fmd_f, Ntot, sign, aval, aPeak, a0chirp, fm_param, beta, nLambda):
    """Tabulate inner A(xi) integrals - fully vectorized on GPU."""
    
    halfStep = 0.5 / nLambda

    xi_a = (torch.arange(Ntot, dtype=DTYPE, device=DEVICE) - 1.0 - Ntot / 2) / nLambda

    # Midpoint xi values for Simpson's rule (shape: Ntot-1)
    xi_mid_alpha = xi_a[1:] - halfStep        # for alpha_x and cosine:  xip - 0.5/nLambda
    xi_mid_fmod  = xi_a[1:] - 0.5    

    # --- ax_a: evaluate at all xi points at once ---
    ax_a = alpha_x(aval, xi_a, sign, iTypeEnv, exp_f, exp_xi) * \
           torch.cos(2.0 * pi * fmod(xi_a, modType, sign, a0chirp, fm_param, beta, fmdfunc, fmd_xi, fmd_f) * xi_a)

    # --- axp_a: evaluate at all midpoints at once ---
    axp_a = alpha_x(aval, xi_mid_alpha, sign, iTypeEnv, exp_f, exp_xi) * \
            torch.cos(2.0 * pi * fmod(xi_mid_fmod, modType, sign, a0chirp, fm_param, beta, fmdfunc, fmd_xi, fmd_f) * xi_mid_alpha)

    # --- Simpson's rule cumulative sum - replaces the entire for loop ---
    increments  = (ax_a[:-1]    + ax_a[1:]    + 4.0 * axp_a)    / 6.0
    increments2 = (ax_a[:-1]**2 + ax_a[1:]**2 + 4.0 * axp_a**2) / 6.0

    axi_a  = torch.zeros(Ntot, dtype=DTYPE, device=DEVICE)
    axi2_a = torch.zeros(Ntot, dtype=DTYPE, device=DEVICE)
    axi_a[1:]  = torch.cumsum(increments,  dim=0)
    axi2_a[1:] = torch.cumsum(increments2, dim=0)

    return ax_a, axi_a / nLambda, axi2_a / nLambda, xi_a

def electronBeamGeneration(batchSize, startIndex = None):
    #-------------------------------------------------------
    # Create or read in initial distribution for the e-beam
    #-------------------------------------------------------
    if (iFile != 1):
        pxv, pyv, pzv, gav, xv, yv = sample_df(batchSize,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p_x,sigma_p_y,a0)
    else:
        # print(f"Rank: [{GLOBAL_RANK}] :: iFile option. Electron-beam generation using values present in File.")
        # print(f"Rank: [{GLOBAL_RANK}] :: StartIndex: {startIndex}, endIndex: {startIndex+batchSize}")
        pxv, pyv, pzv, gav, xv, yv = sample_df_vectors_from_index(startIndex, startIndex+batchSize)

    pxvt = pxv * lm
    pyvt = pyv * lm
    pzvt = pzv * lm

    pv   = torch.sqrt(pxv**2 + pyv**2 + pzv**2)
    be_x = pxv / pv
    be_y = pyv / pv
    be_z = pzv / pv
    be   = torch.sqrt(be_x**2 + be_y**2 + be_z**2)

    prat   = (1.0 - be_x * math.sin(Phi) - be_z * math.cos(Phi)) / (1.0 + be)
    prat_t = prat * torch.sqrt(
                ((pxvt * sig_z) / (prat * sigma_p_x))**2 +
                ((pyvt * sig_z) / (prat * sigma_p_y))**2 + 1.0)

    x0 = pxv / pzv
    y0 = pyv / pzv

    sig_z_v = torch.sqrt(1.0 / (
                pxvt**2 / sigma_p_x**2 +
                pyvt**2 / sigma_p_y**2 +
                prat**2 / sign**2))

    eta_v = sig_z_v**2 * (
                (pxvt * xv) / sigma_p_x**2 +
                (pyvt * yv) / sigma_p_y**2)

    a0_v  = a0 * torch.exp(
                -xv**2 / (2.0 * sigma_p_x**2) -
                yv**2 / (2.0 * sigma_p_y**2)) * \
                torch.exp(eta_v**2 / (2.0 * sig_z_v**2))

    shift_v = ((En0 / mc2) / gav)**2 / prat_t

    return a0_v, shift_v, x0, y0, gav

#==============================================================================
# GPU-ACCELERATED COMPUTATION
#==============================================================================
def compute_spectrum(nLambda, Nout, Ntot,
                         omega_omega0_min, omega_omega0_max, omega0, beta, d_omega,
                         N_MC, rad_v, phi_v, ax_v, axi_v, axi2_v, xi, lambda0, fc, 
                         currentRankNpartBatchSize, sharedMemorySpace: Atomics,mc_batch_size=100, ntot_batch_size=2000,         
                         freq_batch_size=50):
    """
    Compute spectrum for all particles using GPU acceleration.

    Parameters:
    - mc_batch_size: Number of MC angles to process per batch
    - ntot_batch_size: Number of integration points to process per batch
    - freq_batch_size: Number of frequencies to process per batch
    """

    print(f"Rank: [{GLOBAL_RANK}] :: compute_spectrum:  currentRankNpartBatchSize: {currentRankNpartBatchSize}")
    
    omLam_base = 1.0 / (c / lambda0)
    hbar_val = 6.582e-16
    mc2_val = 511e3

    freq_indices = torch.arange(Nout, device=DEVICE, dtype=DTYPE)
    omega_factors = omega0 * (omega_omega0_min + freq_indices * d_omega)  # [Nout]

    results = torch.zeros(Nout, device = DEVICE, dtype = DTYPE) 
    
    isNpartWorkloadLeft,actualSizeOfBatch,counterNpartValue = sharedMemorySpace.anySharedNpartWorkloadLeft(currentRankNpartBatchSize) 
    while (isNpartWorkloadLeft):

        a0_batch, shift_batch, x0_batch, y0_batch, gav_batch = electronBeamGeneration(currentRankNpartBatchSize, TotalNpart - counterNpartValue)

        betaz = torch.sqrt(1.0 - 1.0 / (gav_batch ** 2))
        c1 = gav_batch * (1.0 + betaz)
        part_results = torch.zeros((currentRankNpartBatchSize, Nout), device=DEVICE, dtype=DTYPE)

        for freq_start in range(0, Nout, freq_batch_size):

            freq_end = min(freq_start + freq_batch_size, Nout)
            n_freqs = freq_end - freq_start

            omega_p_batch = omega_factors[freq_start:freq_end][None, :] * shift_batch[:, None]
            f_val_batch = fc * (omega_p_batch * hbar_val / (gav_batch[:, None] * mc2_val))
            omLam_batch = omega_p_batch * omLam_base
            freq_result = torch.zeros((currentRankNpartBatchSize, n_freqs), device=DEVICE, dtype=DTYPE)
                    
            # === BATCH OVER MC ANGLES ===
            for mc_start in range(0, N_MC, mc_batch_size):

                mc_end = min(mc_start + mc_batch_size, N_MC)
                n_mc = mc_end - mc_start

                rad_batch = rad_v[mc_start:mc_end]
                phi_batch = phi_v[mc_start:mc_end]

                x0p = x0_batch[:, None] + rad_batch[None, :] * torch.cos(phi_batch[None, :])
                y0p = y0_batch[:, None] + rad_batch[None, :] * torch.sin(phi_batch[None, :])
                rad_p = torch.sqrt(x0p**2 + y0p**2)

                ct = torch.cos(rad_p)
                ph = torch.atan2(y0p, x0p)

                st = torch.sqrt(1.0 - ct**2)
                cp = torch.cos(ph)
                sp = torch.sin(ph)

                betaz_exp = betaz[:, None, None]
                c1_exp = c1[:, None, None]
                f_val_exp = f_val_batch[:, :, None]
                ct_exp = ct[:, None, :]
                st_exp = st[:, None, :]
                cp_exp = cp[:, None, :]
                sp_exp = sp[:, None, :]

                c0 = (1.0 + betaz_exp) / (1.0 + betaz_exp - f_val_exp * (1.0 + ct_exp))
                c2 = (1.0 - betaz_exp * ct_exp) / (1.0 + betaz_exp)
                c3 = st_exp * cp_exp / c1_exp
                c4 = (1.0 + ct_exp) / (2.0 * (c1_exp ** 2))

                n1x = torch.zeros((currentRankNpartBatchSize, n_freqs, n_mc), device=DEVICE, dtype=DTYPE)
                n2x = torch.zeros((currentRankNpartBatchSize, n_freqs, n_mc), device=DEVICE, dtype=DTYPE)
                n1z = torch.zeros((currentRankNpartBatchSize, n_freqs, n_mc), device=DEVICE, dtype=DTYPE)
                n2z = torch.zeros((currentRankNpartBatchSize, n_freqs, n_mc), device=DEVICE, dtype=DTYPE)

                # === BATCH OVER NTOT (integration points) ===
                for nt_start in range(0, Ntot, ntot_batch_size):
                    nt_end = min(nt_start + ntot_batch_size, Ntot)

                    xi_batch = xi[nt_start:nt_end]
                    ax_batch = ax_v[nt_start:nt_end]
                    axi_batch = axi_v[nt_start:nt_end]
                    axi2_batch = axi2_v[nt_start:nt_end]

                    ax_scaled = a0_batch[:, None] * ax_batch[None, :]
                    axi_scaled = a0_batch[:, None] * axi_batch[None, :]
                    axi2_scaled = (a0_batch[:, None] ** 2) * axi2_batch[None, :]

                    dn1x, dn2x, dn1z, dn2z = _compute_ntot_batch(
                        ax_scaled, axi_scaled, axi2_scaled, xi_batch,
                        omLam_batch, c0, c2, c3, c4)

                    n1x += dn1x
                    n2x += dn2x
                    n1z += dn1z
                    n2z += dn2z
                    
                dE2_val = _compute_final_values(
                    n1x, n2x, n1z, n2z, c1_exp, c2, c0, f_val_exp, ct_exp,
                    betaz_exp, sp_exp, cp_exp, st_exp)

                freq_result += dE2_val.sum(dim=2)
            part_results[:, freq_start:freq_end] = freq_result * ((lambda0 * omega_p_batch / nLambda) ** 2)
                
        results += (part_results[:actualSizeOfBatch] / (shift_batch[:actualSizeOfBatch, None] ** 2)).sum(dim = 0) 

        isNpartWorkloadLeft,actualSizeOfBatch,counterNpartValue = sharedMemorySpace.anySharedNpartWorkloadLeft(currentRankNpartBatchSize)

    return results

#==============================================================================
# Main program helper functions
#==============================================================================

def sample_df(particles,sigma_e_x,sigma_e_y,sigma_e_px,sigma_e_py,gamma,sig_g,pmag0,sigmaPmag,sigma_p_x,sigma_p_y,a0):
    pxv=pmag0*sigma_e_px*torch.randn(particles,dtype=DTYPE,generator=GENERATOR,device=DEVICE)
    pyv=pmag0*sigma_e_py*torch.randn(particles,dtype=DTYPE,generator=GENERATOR,device=DEVICE)
    pzv=torch.sqrt((pmag0+sigmaPmag*torch.randn(particles,dtype=DTYPE,generator=GENERATOR,device=DEVICE))**2-(pxv**2)-(pyv**2))
    gav=torch.sqrt((c**2)*(pxv**2+pyv**2+pzv**2)+(mc2*eV2J)**2)/(mc2*eV2J)
    xv=sigma_e_x*torch.randn(particles,dtype=DTYPE,generator=GENERATOR,device=DEVICE)
    yv=sigma_e_y*torch.randn(particles,dtype=DTYPE,generator=GENERATOR,device=DEVICE)

    return pxv, pyv, pzv, gav, xv, yv

def read_sample_df_file(requestedNpart, filename, device):

    with open(filename,'r') as file:
        npartDataCountStoredInFile = sum(1 for line in file)

    print(f"Rank: [{GLOBAL_RANK}] :: read_sample_df_file DEVICE: {device}.")
    print(f"Rank: [{GLOBAL_RANK}] :: TotalNpart: {TotalNpart}, npart data stored in [XSENSE_Input/ICs_SENSE.in] file: {npartDataCountStoredInFile}.")

    if npartDataCountStoredInFile < requestedNpart:
        print(f"Rank: [{GLOBAL_RANK}] :: Aborting.")
        print(f"Rank: [{GLOBAL_RANK}] :: [XSENSE_Input/ICs_SENSE.in] contains data for only {npartDataCountStoredInFile} particles but the number of particles requested in [XSENSE_Input/config.in] for this simulation were {requestedNpart}.")
        print(f"Rank: [{GLOBAL_RANK}] :: Provide a different [XSENSE_Input/config.in] file with sufficient data to run simulation for {requestedNpart} particles or disable iFile option by specifying [0 # simulation: =1: read ICs from file; <> 1: MC] in [XSENSE_Input/config.in].")
    
        exit()
    
    data = np.loadtxt(filename, max_rows=requestedNpart, dtype= np.float64 if DTYPE == torch.float64 else np.float32)
    data = torch.from_numpy(data).to(device)
    pxv,pyv,pzv,gav,xv,yv = [data[:,i].contiguous() for i in range(6)]

    return pxv, pyv, pzv, gav, xv, yv


def sample_df_vectors_from_index(startIndex, endIndex):
    indexes = torch.arange(startIndex, endIndex).clamp_(max=TotalNpart-1)
    # print(f"Rank: [{GLOBAL_RANK}] :: {indexes}")
    return pxv[indexes],pyv[indexes],pzv[indexes],gav[indexes],xv[indexes],yv[indexes]

def writeToFile(fileName, length, firstColumn, secondColumn, thirdColumn, fourthColumn):
    f = open(fileName,'w')
    for i in range(0, length):
        line = OUTPUT_FILE_LINE_FORMAT % (firstColumn[i],secondColumn[i],thirdColumn[i],fourthColumn[i])
        f.write(line+'\n')
    f.close()

def sample_circle(nmc,Rmax):
    phi=2*torch.pi*torch.rand(nmc,device=DEVICE,generator=GENERATOR,dtype=DTYPE)
    u=torch.rand(nmc,device=DEVICE,generator=GENERATOR,dtype=DTYPE)+torch.rand(nmc,device=DEVICE,generator=GENERATOR,dtype=DTYPE)
    r_val=torch.where(u>1,2-u,u)
    radius=Rmax*r_val

    x=radius*torch.cos(phi)
    y=radius*torch.sin(phi)

    return radius, phi, x, y

def sample_rectangle(nmc,L_aper,x_aper,y_aper):

    x = -0.5*x_aper + x_aper*torch.rand(nmc,device=DEVICE,generator=GENERATOR,dtype=DTYPE)
    y = -0.5*y_aper + y_aper*torch.rand(nmc,device=DEVICE,generator=GENERATOR,dtype=DTYPE)
    radius=torch.atan(torch.sqrt(x**2+y**2)/L_aper)
    phi=torch.where(
        y>0.0,
        torch.where(
            x>0.0,
            torch.atan(y/x),
            pi-torch.atan(y/x)
        ),
        torch.where(
            x<0.0,
            pi+torch.atan(torch.abs(y/x)),
            2*pi-torch.atan(torch.abs(y/x))
        )
    )
    return radius, phi, x, y

def memoryTransferFromGPUToCPU(deviceType: torch.device, dataToBeTransferredFromGPUToCPU):

    if(deviceType.type == CUDA):
        return dataToBeTransferredFromGPUToCPU.cpu().numpy()
    
    return dataToBeTransferredFromGPUToCPU

def normalizeOutput(gatheredResults, Npart, xplot):
    #------------------------------
    # Organize spectrum for output
    #------------------------------
    Yall = memoryTransferFromGPUToCPU(DEVICE,gatheredResults)  # [Nout]
    for i in range(0, Nout):
        if (xplot[i] == 0.0):
            Yall[i] = 0.0
        else:
            Yall[i] = Yall[i]/xplot[i]
    Ynorm = max(Yall)

    yplotN=[]
    yplotNn=[]
    for i in range(0, Nout):
        yplotN.append(yfac*Yall[i]/Npart)
        yplotNn.append(Yall[i]/Ynorm)

    return yplotN,yplotNn

def run_computation(sharedMemorySpace):
        
    return compute_spectrum(
        nLambda, Nout, Ntot,
        wtilde_min, wtilde_max, omega0, beta, d_omega,
        N_MC, rad_v, phi_v, ax, axi, axi2, xi, lambda0, fc, currentRankNpartBatchSize,sharedMemorySpace,mc_batch_size=mc_batch_size, ntot_batch_size=ntot_batch_size,
        freq_batch_size=freq_batch_size
    )

def timeSimulation(isWarmup,iterations,sharedMemorySpace):
    times = []
    
    for i in range(iterations):
        if WORLD_SIZE > 1:
            distCommunication.barrier()
        if GLOBAL_RANK == MASTER_PROCESS_RANK:
            sharedMemorySpace.setupSharedMemoryValue(TotalNpart)
        if WORLD_SIZE > 1:
            distCommunication.barrier()

        if DEVICE.type == CUDA:
            torch.cuda.synchronize()
        start = timeit.default_timer()
        results = run_computation(sharedMemorySpace)
        if DEVICE.type == CUDA:
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    if(isWarmup == True):
        print(f"Rank: [{GLOBAL_RANK}] :: DEVICE: {DEVICE}, WARMUP: TotalNpart: {TotalNpart},Nout: {Nout}, N_MC:{N_MC}, Ntot: {Ntot}, Time: {end-start:.4f} seconds")
    else:
        print(f"Rank: [{GLOBAL_RANK}] :: DEVICE: {DEVICE}, BENCHMARK: TotalNpart: {TotalNpart},Nout: {Nout}, N_MC:{N_MC}, Ntot: {Ntot}")
        print(f"Rank: [{GLOBAL_RANK}] :: DEVICE: {DEVICE}, All times: {[f'{t:.4f}' for t in times]}")
        print(f"Rank: [{GLOBAL_RANK}] :: DEVICE: {DEVICE}, Minimum Time Taken(Benchmark): {min(times)}")
        print(f"Rank: [{GLOBAL_RANK}] :: Dimension of Results: {results.shape}")
    return results

#*****************
# MAIN PROGRAM
#*****************

if __name__ == '__main__':
    
    print(f"Rank: [{GLOBAL_RANK}] :: Device being used: {DEVICE}, Torch compile mode: {TORCH_COMPILE_MODE}")
    currentRankInformation = RankInformation(GLOBAL_RANK, DEVICE)

    if WORLD_SIZE > 1:
        distCommunication = rankCommunication.configureBaseCommunication(WORLD_SIZE)
        ranksInfomationDistribution = RanksInformationDistribution(WORLD_SIZE,currentRankInformation,distCommunication)
        globalKnowledgeRankInformation = ranksInfomationDistribution.retrieveGlobalRanksInformationList()
        rankCommunication.configureCommunicationForGPUsIfTheyArePresent(globalKnowledgeRankInformation)
        print(f"Rank: [{GLOBAL_RANK}] :: cpuRanks: {rankCommunication.getCpuRanks()}, gpuRanks: {rankCommunication.getGpuRanks()}")

    if (iTypeEnv == 1):
        sign = sqrt(2)*sign
        sigma_p_x = sqrt(2)*sigma_p_x
        sigma_p_y = sqrt(2)*sigma_p_y
    elif (iTypeEnv == 2):
        n_gamma = 2*alpha*sign*(a0**2)*pi/3.0
    else:
        n_gamma = 2*alpha*sign*(a0**2)*sqrt(pi)/3.0
        
    Ntot = int(sign*Nt)
    print(f"Rank: [{GLOBAL_RANK}] :: Ntot value: {Ntot}")

    mc_batch_size = N_MC 

    if DEVICE.type == CUDA:
        ntot_batch_size = 300
        freq_batch_size = 50
    else:
        ntot_batch_size = Ntot
        freq_batch_size = Nout  


    batchingInformation = BatchingInformation(freq_batch_size, mc_batch_size)
    currentRankNpartBatchSize = batchingInformation.determineNpartBatchSize(DTYPE, DEVICE)    
    # currentRankNpartBatchSize = 8192   # For debugging purposes    

    print(f"Rank: [{GLOBAL_RANK}] :: TotalNpart: {TotalNpart}, Nout: {Nout}, NMC: {N_MC}, Ntot: {Ntot}")
    print(f"Rank: [{GLOBAL_RANK}] :: Batchsizes npart: {currentRankNpartBatchSize}, nout: {freq_batch_size}, nmc: {mc_batch_size}, ntot: {ntot_batch_size}")

    #--------------IMPORTANT-------------------
    # THIS IS REQUIRED TO MAKE SURE THAT RANDOM VALUES FROM DIFF RANKS DO NOT OVERLAP
    setSeedValueForGenerator()
    #--------------IMPORTANT-------------------

    #--------------------------------
    # Normalization of the spectrum
    #--------------------------------
    
    yfac = (q**2)/(64.0*(pi**3)*eps0*(c**3))
    yfac = yfac/hbarJ
    yfac = yfac/float(N_MC)
    if (TypeAp == 0):
        yfac = pi*(theta_max**2)*yfac
    else:
        yfac = (x_aper*y_aper)*yfac

    #--------------------------------------------
    # Set up x-coordinate (frequencies/energies)
    #--------------------------------------------
    xplot = []
    xplotN = []

    gamma = En0/mc2
    beta = sqrt(1.0-1.0/(gamma**2))
    om0 = (((1+beta)*gamma)**2)*(2*pi*c/lambda0)
    dom = (wtilde_max - wtilde_min)/(Nout-1)
    for i in range(0, Nout):
        xvalN = wtilde_min + dom*i
        xplotN.append(xvalN)
        xval = om0*hbar*xvalN
        xplot.append(xval)
    
    c1 = gamma*(1.0 + beta)
    omega0 = 2.0*pi*c1*c1*(c/lambda0)
    d_omega = (wtilde_max - wtilde_min)/(Nout-1)

    eps_x = eps_x_n/gamma
    eps_y = eps_y_n/gamma
    sigma_e_px = eps_x/sigma_e_x
    sigma_e_py = eps_y/sigma_e_y
    pmag0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigmaPmag = (eV2J/c)*sqrt(((En0*(1.0+sig_g))**2)-mc2**2)-pmag0
    E_laser = 2*pi*hbar*c/lambda0

    Xs = 4.0*E_laser*gamma/(me*(c**2)*J2eV)

    print(f"Rank: [{GLOBAL_RANK}] :: En0 value: {En0}")
    if (TypeAp == 0):
        rad_v, phi_v, xToWrite, yToWrite = sample_circle(N_MC,theta_max)
        writeToFile(ANGLES_SENSE_FILE_PATH, N_MC, rad_v, phi_v, xToWrite, yToWrite)
    else:
        rad_v, phi_v, xToWrite, yToWrite = sample_rectangle(N_MC,L_aper,x_aper,y_aper)
        writeToFile(ANGLES_SENSE_FILE_PATH, N_MC, rad_v, phi_v, xToWrite, yToWrite)

    #-------------------------------------------------------------------
    # Set the resolution for the Simpson 1/3 rule integration
    #-------------------------------------------------------------------
    
    nLambda = 100.0
    #---------------------------------------------------------------
    # Tabulate inner A(xi) integrals (Terzic et al. 2019: Eq. (3)).
    #---------------------------------------------------------------

    ax,axi,axi2,xi = tabulate1_ax(iTypeEnv, exp_f, exp_xi, modType,fmdfunc,fmd_xi,fmd_f,Ntot,sign,1.0,a0,a0chirp,fm_param,beta,nLambda)
    
    sig_z = sign * lambda0
    lm    = lambda0 / (me * c)
    Phi   = pi - l_angle

    #---------------------------
    # Dump the chirping profile
    #---------------------------
    fxi = fmod(xi,modType,sign,a0chirp,fm_param,beta,fmdfunc,fmd_xi,fmd_f)

    #--------------------------------------------
    # Set up regimes of operation
    #--------------------------------------------
    if iCompton == 1:
        fc = 1.0
    else:
        fc = 0.0

    if (iFile == 1):
        pxv, pyv, pzv, gav, xv, yv= read_sample_df_file(TotalNpart, INPUT_FILE, DEVICE)
    
    if GLOBAL_RANK == MASTER_PROCESS_RANK:
        sharedMemorySpace = Atomics("totalNpart",True)
    else:
        sharedMemorySpace = Atomics("totalNpart",False)

    # print(f"Rank: [{GLOBAL_RANK}] :: Warm-up run...")
    print(f"Rank: [{GLOBAL_RANK}] :: Production run...")
    results = timeSimulation(True, 1, sharedMemorySpace)

    # print(f"Rank: [{GLOBAL_RANK}] :: Benchmarking...")
    # if(DEVICE.type == 'cpu'):
    #     results = timeSimulation(False,1,sharedMemorySpace)
    # else:
    #     results = timeSimulation(False,1,sharedMemorySpace)

    print(f"Rank: [{GLOBAL_RANK}] :: Done")

    if WORLD_SIZE > 1:

        distCommunication.barrier()
        if GLOBAL_RANK == MASTER_PROCESS_RANK:
            start = timeit.default_timer()
        gatheredResults = rankCommunication.gatherRankData(results)
        if GLOBAL_RANK == MASTER_PROCESS_RANK:
            end = timeit.default_timer()
            print(f"Rank: [{GLOBAL_RANK}] :: Communication time: {end-start}. Requested Particles: {TotalNpart}")

    sharedMemorySpace.cleanup()
    if GLOBAL_RANK == MASTER_PROCESS_RANK:
        if WORLD_SIZE == 1:
            gatheredResults = results
        
        print(f"Rank: [{GLOBAL_RANK}] :: The complete gathered results shape: {gatheredResults.shape}")
        yplotN,yplotNn = normalizeOutput(gatheredResults,TotalNpart, xplot)
        
        # #-----------------------------------
        # # Dump the total spectrum to a file
        # #-----------------------------------
        writeToFile(OUTPUT_FILE_PATH, Nout, xplot, xplotN, yplotN, yplotNn)

        

