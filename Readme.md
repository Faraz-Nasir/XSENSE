# XSENSE

## eXtended Simulation of Emitted Nonlinear Scattering Events**

### GPU / CPU / hybrid Monte Carlo simulation of nonlinear inverse Compton scattering (ICS) radiation, built on PyTorch.

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Quick Start](#2-quick-start)
3. [Folder Structure](#3-folder-structure)
4. [Installation: Dependencies and Virtual Environment](#4-installation-dependencies-and-virtual-environment)
5. [Running the Simulation](#5-running-the-simulation-benchsh)
6. [Input Files](#6-input-files)
7. [Output Files](#7-output-files)
8. [How It Works](#8-how-it-works-overview)
9. [Precision (Single vs Double)](#9-precision-single-vs-double)
10. [Batch Sizes](#10-batch-sizes-\--hyperparameters)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What This Project Does

XSENSE computes the physics is described in:

- Terzic et al., EPL 126, 12003 (2019)
- Terzic et al. (2021)
- Krafft, PRL 92, 204802 (2004)
- Terzic, Deitrick, Hofler & Krafft, PRL 112, 074801 (2014)

The spectrum is evaluated for a set of electrons (`Npart`) by Monte Carlo sampling of the electron beam and of the detector aperture (`N_MC` angles). The heavy kernels are written in PyTorch, compiled with `torch.compile`, and can run on NVIDIA GPUs, on CPU cores, or on both at the same time.

## 2. Quick Start

1. Create a Python virtual environment.
2. Install the packages listed in `requirements.txt` into that environment.
3. Edit `XSENSE_Input/config.in` to describe your simulation.
4. Make sure the `XSENSE_Output/` folder exists.
5. With the virtual environment still active, run:

```bash
bash Simulate.sh
```

Details for every step are in the sections below. Run everything from the **project root** (the folder that contains `Simulate.sh`), because all paths are relative to it.

## 3. Folder Structure

```text
<project root>/
├── Simulate.sh                      # Entry point: parses options.
│                                    # launches the simulation with torchrun, writes logs
├── threadingEnvVariablesSetup.sh    # Run once per process by torchrun: sets CPU thread
│                                    # counts (hybrid mode), then starts XSENSE/XSENSE.py
├── requirements.txt                 # Python packages / dependencies (see Section 4)
├── README.md                        # This file
│
├── XSENSE_Input/
│   ├── config.in                    # Simulation parameters (see Section 6)
│   ├── ICs_SENSE.in                 # Optional electron-beam initial conditions (iFile = 1)
│   └── ICs_SENSE_in.bkp             # Backup of a large initial-conditions file (200000 particles)
│
├── XSENSE_Output/                   # Results are written here (create before first run)
│
└── XSENSE/                          # Source code
    ├── XSENSE.py                    # Main program: kernels, e-beam generation, spectrum, output
    ├── PhysicalConstants.py         # Constants, PRECISION (DTYPE), env variables, output names
    ├── ReadFile.py                  # Reads config.in and exposes its parameters
    │
    ├── BatchingMechanism/
    │   └── Batching.py              # Chooses the Npart batch size per process
    │
    ├── RankMechanism/
    │   ├── RankCommunication.py             # Device assignment, process groups, result gathering
    │   ├── RankInformation.py               # Small record: rank + device type
    │   └── RanksInformationDistribution.py  # Shares every rank's info with all other ranks
    │
    └── Prototype/
        └── Atomics.py               # Shared-memory atomic counter that hands out particle work
```

**Optional files** read from the project root when selected in `config.in`:

| File | Needed when |
|------|-------------|
| `ICs_SENSE.in` | `iFile = 1` |

**Files created by a run** (in the project root):

| File | Contents |
|------|----------|
| `HardwareMode-<mode>-TotalProcesses-<n>.log` | Standard log output of the run |
| `Error.log` | Standard error of the run |

## 4. Installation: Dependencies and Virtual Environment

`requirements.txt` lists every package (and exact version) the project needs: PyTorch with CUDA 13 builds, Triton, NumPy, SciPy, the `atomics` package, the NVIDIA CUDA runtime libraries, and their supporting packages.

> **You must create a Python virtual environment and install these packages into it _before_ running `Simulate.sh`.** Do not skip this step and do not install into the system Python.

```bash
python3 -m venv xsense-env
source xsense-env/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

If pip cannot find the `+cu130` PyTorch builds, point it at the PyTorch wheel index as well:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Keep the virtual environment **active** when you run `Simulate.sh`. The script calls `torchrun` and `python` from your `PATH`, so they must come from this environment.

**Notes**

- A recent NVIDIA driver compatible with CUDA 13 is needed for GPU runs.
- On ODU clusters, `Simulate.sh` contains commented-out lines (`module load python3/2025.1-py312` and a `crun -p ~/\<Environment path\> torchrun ...` launch command). Uncomment them and use them in place of the plain `torchrun` line if you run there.

## 5. Running the Simulation (`Simulate.sh`)

```bash
bash Simulate.sh [OPTIONS] [HARDWARE_MODE] [NPROC]
```

### Options

| Option | Description |
|--------|-------------|
| `--hardware-mode MODE` | `0` = GPU only, `1` = CPU only, `2` = hybrid CPU + GPU |
| `--nproc COUNT` | Number of parallel processes |
| `-h`, `--help`, `help` | Print usage |

Positional shortcuts: the 1st argument is the hardware mode, the 2nd is `nproc`.

### Hardware modes

| Mode | Name | Behavior |
|------|------|----------|
| `0` | GPU | One process per GPU. `nproc` is capped at the number of GPUs. Default when GPUs are detected (uses all of them). |
| `1` | CPU | A single CPU process. `nproc` is always forced to 1 to avoid oversubscribing the node. Default when no GPU is detected. |
| `2` | Hybrid | One CPU process (rank 0) plus one process per GPU. `nproc` must be between 2 and (number of GPUs + 1). Default is GPUs + 1. Requires at least one GPU. |

### Examples

```bash
bash Simulate.sh                                # Automatic mode and process count
bash Simulate.sh 0 4                            # GPU mode, 4 processes
bash Simulate.sh --hardware-mode 0 --nproc 2    # GPU mode, 2 processes
bash Simulate.sh 1                              # CPU mode
bash Simulate.sh 2                              # Hybrid mode, GPUs + 1 processes
bash Simulate.sh --hardware-mode 2 --nproc 3    # Hybrid mode, 3 processes
bash Simulate.sh --help                         # To see the help page for the script.
```

### What `Simulate.sh` does

1. Detects the GPUs using `nvidia-smi` and validates your mode / process count.
2. Exports `HARDWARE_MODE`.
3. For each `Npart` value, rewrites **line 28** (the Npart line) of `XSENSE_Input/config.in` **in place**.
4. Launches `torchrun` on one node, which starts `threadingEnvVariablesSetup.sh` once per process; that script then runs `XSENSE/XSENSE.py`.
5. Appends output to `HardwareMode-<mode>-TotalProcesses-<n>.log` and errors to `Error.log`.

> **Important**
>
> - `bench.sh`, The bash script used for **dedbugging** purposes, **overwrites** the Npart value in `config.in`. The sweep is set by the `NPART_VALS` line near the bottom of `bench.sh`. At the moment it is `for(i=20000; i<=20000; i+=10000)`, i.e. a single value, `Npart = 20000`, used for debugging. Change the start, end and step to run your own sweep, for example `i=10000; i<=100000; i+=10000`.
> - `bench.sh` pauses 5 seconds between Npart values.
> - The first run includes `torch.compile` compilation (GPU processes use the `max-autotune` mode), so the warm-up can take a few minutes.

## 6. Input Files

### 6.1 `XSENSE_Input/config.in`

Each line holds one number followed by a space and a comment. The value **must be followed by a space**, because the parser reads up to the first space. Do not add, remove or reorder lines. The line number is the parameter's position:

| Line | Parameter |
|-----:|-----------|
| 1 | e-beam mean energy `En0` [eV] |
| 2 | e-beam relative energy spread |
| 3 | e-beam horizontal size `sigma_e_x` [m] |
| 4 | e-beam vertical size `sigma_e_y` [m] |
| 5 | e-beam horizontal normalized emittance [m rad] |
| 6 | e-beam vertical normalized emittance [m rad] |
| 7 | laser wavelength `lambda0` [m] |
| 8 | number of wavelengths in the laser pulse (called `sign` in the code) |
| 9 | laser horizontal spot size `sigma_p_x` [m] |
| 10 | laser vertical spot size `sigma_p_y` [m] |
| 11 | laser field strength parameter `a0` |
| 12 | laser envelope `iTypeEnv` (1 = Gaussian, 2 = rectangular) |
| 13 | laser chirping type `modType` |
| 14 | chirp parameter (meaning depends on `modType`: `a0chirp`, `lambda_RF` or `p`) |
| 15 | laser horizontal crossing angle |
| 16 | laser vertical crossing angle |
| 17 | polarization |
| 18 | aperture type `TypeAp` (0 = circular, 1 = rectangular) |
| 19 | distance from interaction point to aperture `L_aper` [m] |
| 20 | `R_aper` [m] if circular, `x_aper` [m] if rectangular |
| 21 | ignored if circular, `y_aper` [m] if rectangular |
| 22 | aperture offset in x |
| 23 | aperture offset in y |
| 24 | start of the spectrum|
| 25 | end of the spectrum|
| 26 | `Nout`: number of output points in the spectrum |
| 27 | `Ntot`: integration points per laser wavelength |
| 28 | `Npart`: total number of electrons (`TotalNpart`). |
| 29 | `N_MC`: number of Monte Carlo aperture angles per electron |
| 30 | `iFile`: 1 = read electron initial conditions from `ICs_SENSE.in`, 0 = generate particle data on the fly. |
| 31 | subdivisions in theta integration |
| 32 | subdivisions in phi integration |
| 33 | `iCompton`: 1 = Compton, any other value = Thomson |

### 6.2 `XSENSE_Input/ICs_SENSE.in` (only used when `iFile = 1`)

Plain text, one particle per line, 6 columns:

```text
px   py   pz   gamma   x   y
```

The file must contain **at least `Npart (value from config file)` lines**. If it contains fewer, the program prints an error and exits. To use a larger set, copy `ICs_SENSE_in.bkp` (200000 particles) over `ICs_SENSE.in` or instead set `iFile = 0` to generate data on the fly. Every process loads Npart worth of data from the file into its own memory (CPU RAM, or GPU memory for GPU processes).

## 7. Output Files

**`XSENSE_Output/XSENSE:<GPU|CPU|CPU+GPU>:<processes>:TotalNpart:<Npart>.txt`**

Written by the master process (global rank 0). 

**`XSENSE_Output/angles_SENSE:<node rank>:<local rank>.txt`**

The `N_MC` sampled aperture angles used by each process (radius, phi, x, y).

## 8. How It Works (Overview)

### Process layout

`Simulate.sh` calls `torchrun`, which starts one Python process (rank) per device. Each rank decides its device:

- **GPU mode:** rank *i* uses GPU *i*.
- **CPU mode:** the single rank uses the CPU.
- **Hybrid mode:** global rank 0 uses the CPU and the remaining ranks use GPUs 0 to n-2.

Do not change that rank ordering; the result gathering depends on it.

### Dynamic work sharing

A shared-memory counter (`Atomics.py`) is initialized by rank 0 with `Npart`. Each process repeatedly subtracts its own Npart batch size from the counter (an atomic operation) and receives that slice of particles to process. Devices with larger memory capacity take more work than others and nobody sits idle. When the counter reaches zero the process is finished.

### Per-batch computation (`compute_spectrum` in `XSENSE.py`)

1. Build the electron beam for the batch, either by random sampling or by reading the matching rows of `ICs_SENSE.in`.
2. Loop over frequency chunks (`Nout`), Monte Carlo angle chunks (`N_MC`) and integration chunks (`Ntot`).
3. The innermost kernel (cos / sin / sum over the `Ntot` integration points) is compiled with `torch.compile` and accumulates the radiation integrals.
4. Combine the integrals into the spectrum contribution of each electron and sum over the electrons in the batch.

### Result gathering

GPU ranks send their partial spectra to a GPU aggregator rank over NCCL. The aggregator and any CPU rank then send their data to the master rank over Gloo. The master sums everything, normalizes it, and writes the output file.

### Random numbers

Each rank seeds its own generator with `1000 + global rank`, so ranks never produce the same random samples.

## 9. Precision (Single vs Double)

The numerical precision is set in **one place**:

| | |
|---|---|
| **File** | `XSENSE/PhysicalConstants.py` |
| **Line** | `DTYPE = torch.float64 \| torch.float32` |

| Value | Meaning |
|-------|---------|
| `torch.float64` | Double precision (default). Most accurate. |
| `torch.float32` | Single precision. Faster and uses half the memory, which allows larger batches. Hardware with a weak FP64 rate (for example consumer GPUs) benefits the most. |

Change that single line, then run `Simulate.sh` again. Everything else follows from it: tensors, the electron input file read, communication buffers, and the memory estimate used for the Npart batch size.

**Guidance:** Precision is chosen by the user. Double precision is the safe choice, particularly for small `Npart`. Single precision can be worthwhile for large `Npart`. Before relying on single precision, compare a single-precision spectrum against a double-precision one for your own parameters (`Npart`, `Ntot`, `N_MC`, `Nout`), because the acceptable difference can depend on them.

## 10. Batch Sizes - Hyperparameters

The work is split into batches along four dimensions so that the large intermediate tensors fit in memory.

| Dimension | Meaning | Where it is set |
|-----------|---------|-----------------|
| `Npart` | Particles processed at once | **Automatically** `XSENSE/BatchingMechanism/Batching.py` |
| `Nout` | Frequency points per chunk | `XSENSE/XSENSE.py`, main block: `freq_batch_size` |
| `N_MC` | Aperture angles per chunk | `XSENSE/XSENSE.py`, main block: `mc_batch_size` |
| `Ntot` | Integration points per chunk | `XSENSE/XSENSE.py`, main block: `ntot_batch_size` |

### 10.1 Nout, N_MC and Ntot batch sizes (`XSENSE/XSENSE.py`, main block)

| Setting | GPU processes | CPU processes |
|---------|---------------|---------------|
| `ntot_batch_size` | 300 | `Ntot` (unbatched) |
| `freq_batch_size` | 50 | `Nout` (unbatched) |
| `mc_batch_size` | `N_MC` (all angles in one chunk) | `N_MC` (unbatched) |

Edit those lines to change them. Smaller values use less memory but run more loop iterations; larger values use more memory. Changing them also changes the Npart batch size.

### 10.2 Npart batch size (automatic, `XSENSE/BatchingMechanism/Batching.py`)

**GPU processes**

1. Reads the free GPU memory and uses it as the budget.
2. Estimates bytes per particle
3. Rounds down to a power of two, never above the smallest power of two that is >= `Npart`.
4. Verifies by actually allocating test tensors and halves the size on an out-of-memory error.
5. The result is printed in the log as `Npart batch size: ...`.

**CPU processes**

The Npart batch size equals the number of threads of that process (`torch.get_num_threads()`). In hybrid mode, `threadingEnvVariablesSetup.sh` gives each GPU process 1 thread and gives the CPU process all remaining threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`). If no threads are left for the CPU process, the run stops with a message; use GPU mode or fewer GPUs.

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `No such file or directory` for `XSENSE_Input/...` or `XSENSE_Output/...` | You are not in the project root, or `XSENSE_Output/` has not been created. |
| `torchrun: command not found` | The virtual environment is not active. |
| `GPU Mode (0) requested, but no GPUs were detected` | Use mode 1, or check `nvidia-smi` and your driver. |
| Out-of-memory on a GPU | Raise `safety_factor`, lower the `0.60` memory fraction in `Batching.py`, or lower `freq_batch_size` / `ntot_batch_size` in `XSENSE.py`. |
| `ICs_SENSE.in contains data for only N particles` | Use a larger initial-conditions file (see `ICs_SENSE_in.bkp`), lower `Npart`, or set `iFile` to 0. |
| Run seems stuck at warm-up | The first GPU run compiles and autotunes kernels, which can take a while. |
|                            | To evaluate benchmark times for the simulation. Run a warmup call and then a benchmark call for the compuation. |
| Leftover shared memory counter after a crashed run | The code removes stale shared memory on the next start. If it persists, delete `/dev/shm/totalNpart`. |
| Any failed run | Check `Error.log` for details. |
