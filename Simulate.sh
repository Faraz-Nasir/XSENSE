#!/bin/bash

help(){
    cat << EOF
    [NAME]  XSENSE Simulation
    
    [USAGE] bash Simulate.sh [OPTIONS] [HARDWARE_MODE] [NPROC]

    [OPTIONS]
        --hardware-mode     MODE                    Set hardware mode: 0 (GPU Mode), 1 (CPU Mode), 2 (Hybird Mode).
        --nproc             PROCESS_COUNT           Set the number of parallel processes.
        -h, --help, help 
    
    [POSITIONAL ARGUMENTS]
        HARDWARE_MODE                               1st positional argument ( 0, 1, or 2 )
        NPROC                                       2nd positional argument ( Number of parallel processes.)

    [EXAMPLES]
        bash Simulate.sh                               Automatically configures hardware-mode and number of processes.
                                                    Defaults to GPU Mode.
        
        bash Simulate.sh --hardware-mode 0 --nproc 2   
        bash Simulate.sh 0 4
        
        bash Simulate.sh --hardware-mode 1             CPU Mode (nproc defaults to 1, to prevent oversubscription on the node.)
        bash Simulate.sh 1                             CPU Mode (nproc defaults to 1)
        bash Simulate.sh 1 4                           CPU Mode (nproc is overriden to 1, to prevent oversubscription on the node.)

        bash Simulate.sh 2                             Hybrid Mode (nproc defaults to number of GPUs + 1)
        bash Simulate.sh --hardware-mode 2 --nproc 3   Hybrid Mode (nproc = 3)
EOF
exit 0
}

hardwareModeAndProcessCount(){
    local gpuCount="$1"

    if [ "$gpuCount" -eq 0 ]; then
        echo "1 1"
    else
        echo "0 $gpuCount"
    fi
}

checkGPUAndReturnCount(){
    if command -v nvidia-smi &> /dev/null; then
        local count=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
        echo "${count:-0}"
    else
        echo "0"
    fi
}

userInput(){
    # --- Parsing user parameters ---
    local CLI_HARDWARE_MODE=""
    local CLI_NUMBER_OF_PROCESSES=""

    while [[ "$#" -gt 0 ]]; do
        case $1 in 
            -h|--help|help)
                help
                ;;

            --hardware-mode)
                if [[ -z "$2" || "$2" =~ ^-- || ! "$2" =~ ^[0-9]+$ || "$2" -lt 0 || "$2" -gt 2 ]]; then
                    echo "Error: --hardware-mode requires 0 (GPU Mode), 1 (CPU Mode), 2 (Hybrid Mode)."
                    exit 1
                fi
                CLI_HARDWARE_MODE="$2"
                shift 2
                ;;

            --nproc)
                if [[ -z "$2" || "$2" =~ ^-- || ! "$2" =~ ^[0-9]+$ || "$2" -le 0 ]]; then
                    echo "Error: --nproc invalid value provided."
                    exit 1
                fi
                CLI_NUMBER_OF_PROCESSES="$2"
                shift 2
                ;;

            -*)
                echo "Unknown parameter: $1"
                help
                ;;
            *)
                if [[ -z "$CLI_HARDWARE_MODE" ]]; then
                    if [[ ! "$1" =~ ^[0-9]+$ || "$1" -lt 0 || "$1" -gt 2 ]]; then
                        echo "Error: Invalid hardware mode '$1'. Requires [0,1,2]."
                        exit 1
                    fi
                    CLI_HARDWARE_MODE="$1"
                    shift 1
                elif [[ -z "$CLI_NUMBER_OF_PROCESSES" ]]; then
                    if [[ ! "$1" =~ ^[0-9]+$ ]] || [[ "$1" -le 0 ]]; then
                        echo "Error: Invalid process count '$1'."
                        exit 1
                    fi
                    CLI_NUMBER_OF_PROCESSES="$1"
                    shift 1
                else
                    echo "Too many arguments provided: $1"
                    help
                fi
                ;;
        esac
    done

    local gpuCount=$(checkGPUAndReturnCount)
    local autoHardwareMode autoNumberOfProcesses
    read autoHardwareMode autoNumberOfProcesses < <(hardwareModeAndProcessCount "$gpuCount")

    HARDWARE_MODE="${CLI_HARDWARE_MODE:-$autoHardwareMode}"

    # CPU Mode
    if [ "$HARDWARE_MODE" -eq 1 ]; then
        if [[ -n "$CLI_NUMBER_OF_PROCESSES" && "$CLI_NUMBER_OF_PROCESSES" -ne 1 ]]; then
            echo "Note: Hardware mode 1 (CPU Mode) selected. nproc overriden to 1 to prevent oversubscription."
        fi
        NUMBER_OF_PROCESSES=1
    
    # Hybrid Mode
    elif [ "$HARDWARE_MODE" -eq 2 ]; then

        if [ "$gpuCount" -eq 0 ]; then
            echo "Error: Hybrid mode requires atleast 1 GPU."
            exit 1
        fi 

        local maximumNumberOfProcessesInHybridMode=$(( gpuCount + 1 ))

        if [[ -n "$CLI_NUMBER_OF_PROCESSES" ]]; then
            if [[ "$CLI_NUMBER_OF_PROCESSES" -le 1 || "$CLI_NUMBER_OF_PROCESSES" -gt "$maximumNumberOfProcessesInHybridMode" ]]; then
                echo "Note: Hybrid Mode (2) allows [2,$maximumNumberOfProcessesInHybridMode] processes i.e. Number of GPUs + 1 (CPU process)."
                CLI_NUMBER_OF_PROCESSES="$maximumNumberOfProcessesInHybridMode" 
                echo "Note: nproc overriden to $CLI_NUMBER_OF_PROCESSES"
            fi
            NUMBER_OF_PROCESSES="$CLI_NUMBER_OF_PROCESSES"
        else
            NUMBER_OF_PROCESSES="$maximumNumberOfProcessesInHybridMode"
        fi
    
    # GPU Mode
    else
        if [ "$gpuCount" -eq 0 ]; then
            echo "Error: GPU Mode (0) requested, but no GPUs were detected on the node."
            exit 1
        fi

        if [[ -n "$CLI_NUMBER_OF_PROCESSES" ]]; then
            if [ "$CLI_NUMBER_OF_PROCESSES" -gt "$gpuCount" ]; then
                echo "Note: GPU Mode (0) allows a maximum of $gpuCount processes based on the number of GPUs available on the node."
                CLI_NUMBER_OF_PROCESSES="$gpuCount"
                echo "Note: nproc overriden to $CLI_NUMBER_OF_PROCESSES"
            fi
            NUMBER_OF_PROCESSES="$CLI_NUMBER_OF_PROCESSES"
        else
            NUMBER_OF_PROCESSES="$gpuCount"
        fi
    fi
}

runSimulation() {
    local fileName="HardwareMode-$HARDWARE_MODE-TotalProcesses-$NUMBER_OF_PROCESSES.log"
    local errorFileName="Error.log"
    
    echo "Total Nodes: 1, Node Rank: 0, Master Addr: localhost"
    
    echo "=================================NPART==========================================" | tee -a "$fileName" "$errorFileName" > /dev/null

    # For ODU clusters
    # crun -p ~/envs/GPU_Code torchrun --nproc-per-node=$NUMBER_OF_PROCESSES --nnodes=1 --node-rank=0 --master-addr=localhost --no-python ./threadingEnvVariablesSetup.sh $(nproc) >> "$fileName" 2>> "$errorFileName"

    torchrun --nproc-per-node=$NUMBER_OF_PROCESSES --nnodes=1 --node-rank=0 --master-addr=localhost --no-python ./threadingEnvVariablesSetup.sh $(nproc) >> "$fileName" 2>> "$errorFileName"
    echo "===================================================================================" | tee -a "$fileName" "$errorFileName" > /dev/null

}

# Required for ODU clusters
# module load python3/2025.1-py312

userInput "$@"

echo "Hardware mode: $HARDWARE_MODE, Number of processes: $NUMBER_OF_PROCESSES"
export HARDWARE_MODE

runSimulation