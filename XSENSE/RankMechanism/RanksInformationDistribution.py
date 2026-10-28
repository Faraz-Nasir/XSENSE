from RankMechanism.RankInformation import RankInformation
import torch.distributed as dist
from PhysicalConstants import MASTER_PROCESS_RANK, GLOBAL_RANK

class RanksInformationDistribution:
    _worldSize : int
    _currentRankInformation : RankInformation
    _ranksInformationList : list[RankInformation]= [None]
    _distributionChannel : dist

    def __init__(self, worldSize, currentRankInformation : RankInformation , distributionChannel):
        self._worldSize = worldSize
        self._ranksInformationList = [None] * worldSize
        self._currentRankInformation = currentRankInformation
        self._distributionChannel = distributionChannel

    def printRankInformation(self):
        for rank in self.getRanksInformationList():
            print(f"Rank: [{GLOBAL_RANK}] :: {rank}")

    def retrieveGlobalRanksInformationList(self, masterRank = MASTER_PROCESS_RANK) -> list[RankInformation]:
        
        self._gatherRankInformationList(masterRank)
        self._broadcastCompleteRanksInformationList(masterRank)

        return self.getRanksInformationList()

    def _gatherRankInformationList(self, destinationRank):

        globalRank = self.getCurrentRankInformation().getGlobalRank()

        if globalRank == destinationRank:
            self._recieveRankInformationDetails()          
            print(f"Rank: [{GLOBAL_RANK}] :: Recieved Information from all the ranks.")

        else:
            self._sendCurrentRankInformationDetails(destinationRank)

    def _recieveRankInformationDetails(self):

        currentRankInformation = self.getCurrentRankInformation()
        self._setRankInRanksInformationList(currentRankInformation)

        for rank in range(1,self.getWorldSize()):

            rankInformationReceived : RankInformation = [None]
            self.getDistributionChannel().recv_object_list(rankInformationReceived, src = rank)
            self._setRankInRanksInformationList(rankInformationReceived[0])

    def _sendCurrentRankInformationDetails(self, destinationRank):
        self.getDistributionChannel().send_object_list([self.getCurrentRankInformation()], dst = destinationRank)

    def _broadcastCompleteRanksInformationList(self, masterRank = MASTER_PROCESS_RANK):

        currentRank = self.getCurrentRankInformation().getGlobalRank()

        if currentRank == masterRank :
            payload = self.getRanksInformationList() 
        else: 
            payload = [None] * self.getWorldSize()

        self.getDistributionChannel().broadcast_object_list(payload, src=masterRank)
        self._setRanksInformationList(payload)
    
    def getWorldSize(self):
        return self._worldSize

    def getCurrentRankInformation(self) -> RankInformation:
        return self._currentRankInformation

    def getRanksInformationList(self) -> list[RankInformation]:
        return self._ranksInformationList

    def getRanksInformationListAtIndex(self, index) -> RankInformation:
            return self._ranksInformationList[index]

    def getDistributionChannel(self) -> dist:
        return self._distributionChannel

    def _setRanksInformationList(self, completeRanksInformationList: list[RankInformation]):
        self._ranksInformationList = completeRanksInformationList

    def _setRankInRanksInformationList(self, rankInformation: RankInformation):
        index = rankInformation.getGlobalRank()
        self._ranksInformationList[index] = rankInformation