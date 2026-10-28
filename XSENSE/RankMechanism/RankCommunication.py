import torch
import torch.distributed as dist
from datetime import timedelta

from ReadFile import Nout
from PhysicalConstants import *
from RankMechanism.RankInformation import RankInformation

class RankCommunication:
    _localRank: int
    _globalRank: int
    _hardwareMode :int
    _device: torch.device
    _gpuCommunicationGroup = None 
    _gpuRanks : list
    _cpuRanks : list
    _gpuAggregatorRank : int 

    def __init__(self, localRank, globalRank, hardwareMode):
        self._localRank = localRank
        self._globalRank = globalRank
        self._hardwareMode = hardwareMode
        self._gpuCommunicationGroup = None
        self._gpuAggregatorRank = None
        self._gpuRanks = []
        self._cpuRanks = []

        self._setDevice()

    def configureBaseCommunication(self, worldSize) -> dist:

        if worldSize == 1:
            return

        if self.getDevice().type == CUDA:
            self._assignRankToGPU()

        dist.init_process_group(
            backend = BACKEND_CPU,
            init_method = "env://",
            rank = self.getGlobalRank(),
            world_size = worldSize,
            timeout=timedelta(hours=3),
        )

        return dist

    def configureCommunicationForGPUsIfTheyArePresent(self, ranksInformationList: list[RankInformation]):

        self._identifyDeviceTypeForEveryRank(ranksInformationList)
        gpuRanks = self.getGpuRanks()

        if self.areGPURanksPresent(): 
            gpuCommunicationGroup = dist.new_group(ranks = gpuRanks, backend = BACKEND_GPU)
            self._setGpuCommunicationGroup(gpuCommunicationGroup)

    def areGPURanksPresent(self) -> bool:
        return len(self.getGpuRanks()) > 0

    def gatherRankData(self, resultsTensor: torch.Tensor, masterRank = MASTER_PROCESS_RANK):
        myRank = self.getGlobalRank()
        device = self.getDevice()
        gpuRanks = self.getGpuRanks()
        cpuRanks = self.getCpuRanks()
        aggregatorRank = self.getGpuAggregatorRank()
        resultCollectionDictionary = {}

        if len(gpuRanks) > 0:

            if myRank == aggregatorRank:
                chunks = {}
                for rank in gpuRanks:

                    if rank == myRank:
                        chunks[rank] = resultsTensor
                    else:
                        chunk = self._recieveData(rank, self.getGpuCommunicationGroup(), device)
                        chunks[rank] = chunk

                orderedChunks = [chunks[rank] for rank in gpuRanks]
                aggregatedGpuTensor = torch.stack(orderedChunks, dim = 0).sum(dim=0)

                if myRank == masterRank:
                    resultCollectionDictionary["gpu"] = aggregatedGpuTensor

                else:
                    self._sendData(masterRank, None, aggregatedGpuTensor.cpu())

            elif myRank in gpuRanks:
                self._sendData(aggregatorRank, self.getGpuCommunicationGroup(), resultsTensor)

            if myRank == masterRank and aggregatorRank != masterRank:
                aggregatedGpuTensor = self._recieveData(aggregatorRank, None, torch.device(CPU))
                resultCollectionDictionary["gpu"] = aggregatedGpuTensor

        if len(cpuRanks) > 0:

            if myRank == masterRank:
                for rank in cpuRanks:
                    if rank == masterRank:
                        resultCollectionDictionary[rank] = resultsTensor
                    else:
                        resultCollectionDictionary[rank] = self._recieveData(rank, None, device)

            elif myRank in cpuRanks:
                self._sendData(masterRank, None, resultsTensor)

        from functools import reduce

        return reduce(torch.add, resultCollectionDictionary.values()) if myRank == masterRank else None
    
    def _sendData(self, destination, group, dataTensor : torch.Tensor):

        dist.send(dataTensor, dst = destination, group = group)

    def _recieveData(self, source, group, device, dtype = DTYPE) ->  torch.Tensor:

        dataBuffer = torch.zeros((Nout), dtype = dtype, device = device)
        dist.recv(dataBuffer, src = source, group = group)

        return dataBuffer

    def _identifyDeviceTypeForEveryRank(self, ranksInformationList: list[RankInformation]):

        cpuRanks = []
        gpuRanks = []

        for rank in ranksInformationList:
            if rank.isCPU():
                cpuRanks.append(rank.getGlobalRank())
            else:
                gpuRanks.append(rank.getGlobalRank())

        self._setCpuRanks(cpuRanks)
        self._setGpuRanks(gpuRanks)

        if self.areGPURanksPresent():
            self._setGpuAggregatorRank(min(gpuRanks))

    def _determineDevice(self) -> torch.device:
        '''
            This method checks if GPU's are available to run the simulation.
        '''
        hardwareMode = self.getHardwareMode()
        localRank = self.getLocalRank()

        if hardwareMode == 1:

            print(f"Rank: [{GLOBAL_RANK}] :: CPU - Only mode selected.")
            device = self._assignRankToCPUForTensorCreation("CPU - Only mode", CPU)

        elif hardwareMode == 0:

            print(f"Rank: [{GLOBAL_RANK}] :: GPU - Only mode selected.")

            if torch.cuda.is_available():
                cudaIdx = localRank
                device = self._assignRankToGPUForTensorCreation(cudaIdx, "GPU - Only mode selected", CUDA)
            else:
                print(f"Rank: [{GLOBAL_RANK}] :: CUDA not available, falling back to CPU")
                device = self._assignRankToCPUForTensorCreation("GPU - Only mode", CPU)

        elif hardwareMode == 2:

            '''
                IN MIXED MODE LOCAL RANK 0 IS ALWAYS ASSIGNED TO CPU.
                DON'T CHANGE THIS BECAUSE GATHER RESULTS DEPENDS ON IT.
            '''

            print(f"Rank: [{GLOBAL_RANK}] :: CPU + GPU - Mixed mode selected.")

            if localRank == 0:
                device = self._assignRankToCPUForTensorCreation("Mixed mode", CPU)
            else:
                cudaIdx = localRank - 1
                device = self._assignRankToGPUForTensorCreation(cudaIdx, "Mixed mode", CUDA)

        else:
            raise ValueError(f"Unsupported hardware mode provided : {hardwareMode}")

        return device

    def _assignRankToCPUForTensorCreation(self ,mode ,cpu) -> torch.device:
        device = torch.device(cpu)
        print(f"Rank: [{GLOBAL_RANK}] :: Global Rank: {self.getGlobalRank()} assigned to CPU: ({mode}).")

        return device

    def _assignRankToGPUForTensorCreation(self, cudaDeviceId, mode, cuda) -> torch.device:

        device = torch.device(f"{cuda}:{cudaDeviceId}")
        print(f"Rank: [{GLOBAL_RANK}] :: Local Rank: {self.getLocalRank()} assigned to GPU:{cudaDeviceId} ({mode}).")

        return device

    def _assignRankToGPU(self):
        torch.cuda.set_device(device=self.getDevice())

    def getLocalRank(self):
        return self._localRank
    
    def getGlobalRank(self):
        return self._globalRank

    def getHardwareMode(self):
        return self._hardwareMode

    def getDevice(self) -> torch.device:
        return self._device

    def getGpuAggregatorRank(self):
        return self._gpuAggregatorRank
    
    def getCpuRanks(self):
        return self._cpuRanks

    def getGpuRanks(self):
        return self._gpuRanks

    def getGpuCommunicationGroup(self):
        return self._gpuCommunicationGroup

    def _setDevice(self):
        device = self._determineDevice()
        self._device = device

    def _setGpuCommunicationGroup(self, gpuGroup):
        self._gpuCommunicationGroup = gpuGroup

    def _setGpuRanks(self, gpuRanks):
        self._gpuRanks = gpuRanks

    def _setCpuRanks(self, cpuRanks): 
        self._cpuRanks = cpuRanks

    def _setGpuAggregatorRank(self, gpuAggregatorRank):
        self._gpuAggregatorRank = gpuAggregatorRank


