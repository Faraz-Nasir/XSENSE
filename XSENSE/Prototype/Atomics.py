from multiprocessing import shared_memory
import atomics
from PhysicalConstants import GLOBAL_RANK, MASTER_PROCESS_RANK

class Atomics:
    _sharedHandle : shared_memory.SharedMemory
    _sharedMemoryName: str

    def __init__(self, sharedMemoryName, createSpace, sizeOfSpace = 4):

        self._sharedMemoryName = sharedMemoryName
        self._createSharedHandle(createSpace, sizeOfSpace)

    def setupSharedMemoryValue(self, initialTotalNpartValue):
        
        with atomics.atomicview(buffer = self._getSharedHandle().buf, atype = atomics.INT) as a:
            a.store(initialTotalNpartValue)

    def anySharedNpartWorkloadLeft(self, rankNpartBatchSize) -> tuple[bool, int, int]:  
      
        with atomics.atomicview(buffer = self._getSharedHandle().buf, atype = atomics.INT) as a :
            before = a.fetch_sub(rankNpartBatchSize) # Subtracts the value provided and returns the value before subtraction.

        if before <= 0:
            return (False,0,before)
        else:
            return (True,min(before, rankNpartBatchSize),before)

    def _createSharedHandle(self,createSpace: bool, sizeOfSpace: int):

            if createSpace:
                self.cleanupStaleHandleIfExists()

            self._setSharedHandle(shared_memory.SharedMemory(name = self._getSharedMemoryName(), create = createSpace, size = sizeOfSpace))
            

    def cleanup(self, globalRank = GLOBAL_RANK):

        try:
            self._getSharedHandle().close()
            if globalRank == MASTER_PROCESS_RANK:
                self._getSharedHandle().unlink()
        except FileNotFoundError:
            pass

    def cleanupStaleHandleIfExists(self):
        try:
            staleHandle = shared_memory.SharedMemory(name = self._getSharedMemoryName(), create = False)
            staleHandle.close()
            staleHandle.unlink()
        except FileNotFoundError:
            pass

    def _getSharedHandle(self):
        return self._sharedHandle

    def _getSharedMemoryName(self):
        return self._sharedMemoryName

    def _setSharedHandle(self, sharedHandle):
        self._sharedHandle = sharedHandle

        

