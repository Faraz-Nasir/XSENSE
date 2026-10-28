from PhysicalConstants import CPU
class RankInformation:
    _globalRank_ = 0
    _deviceType = None

    def __init__(self, globalRank, deviceType):
        self._globalRank_ = globalRank
        self._deviceType = deviceType

    def isCPU(self):
        return str(self._deviceType) == CPU
    
    def getGlobalRank(self):
        return self._globalRank_

    def getDeviceType(self):
        return self._deviceType
    
    def __str__(self):
        
        string = (
            f" Global rank: {self._globalRank_} |"
            f" Device type: {self._deviceType} "    
        )
        return string

    __repr__ = __str__
