import torch
from PhysicalConstants import CPU,GLOBAL_RANK
from ReadFile import TotalNpart

class BatchingInformation:
    _CPU = CPU
    _freqBatchSize : int
    _mcBatchSize : int

    def __init__(self, freqBatchSize, mcBatchSize):
        self._freqBatchSize = freqBatchSize
        self._mcBatchSize = mcBatchSize
    
    def getFreqBatchSize(self):
        return self._freqBatchSize

    def getMCBatchSize(self):
        return self._mcBatchSize
    
    def determineNpartBatchSize(self, dtype: torch.dtype, device: torch.device, safety_factor=2.5) -> int:

        if device.type == CPU:
            
            print(f"Rank: [{GLOBAL_RANK}] :: Number of threads for this process: {torch.get_num_threads()}.")
            batchSize = torch.get_num_threads()
        else:
            batchSize = self._determineCudaNpartBatchSize(dtype, device, safety_factor)

        return batchSize

    def _determineCudaNpartBatchSize (self, dtype: torch.dtype, device: torch.device, safety_factor) -> int:

        free_bytes, _ = torch.cuda.mem_get_info()
        
        bytes_per_particle = (5 * self._freqBatchSize * self._mcBatchSize + 3 * self._mcBatchSize + self._freqBatchSize) * dtype.itemsize * safety_factor
        
        budget = int(free_bytes * 0.60)
        batch  = max(1, budget // bytes_per_particle)
        p = 1

        while p < TotalNpart:
            p *= 2
        while p > batch:
            p //= 2

        while p >= 1:
            try:
                probe = [
                    torch.empty((p, self._freqBatchSize, self._mcBatchSize), dtype = dtype, device = device)
                    for _ in range(5)
                ]
                print(f"Rank: [{GLOBAL_RANK}] :: Npart batch size: {p}")
                del probe
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                p //= 2

        return max(1, p) 
