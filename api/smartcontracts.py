import os
import re
import os
import sys
import time
import signal
import random
import warnings
import numpy as np
import pandas as pd
from eth_abi import *
from web3 import Web3
from termcolor import colored
import matplotlib.pyplot as plt 
from subprocess import Popen, PIPE
from eth_account import Account, messages
from web3.exceptions import ContractLogicError
from pytorch_model import gb, rb, b, green, red


warnings.filterwarnings("ignore")

ONLY_PRINT_ROUND_SUMMARY = False
WAIT_DELAY = int(86400*2)


class NotEnoughUnlockedAccounts(Exception):
    pass

def _print(string, end= ""):
    if ONLY_PRINT_ROUND_SUMMARY:
        try:
            print(string.split(":")[0]+ string.split(":")[1].split("|")[0] +
                  "                                                              ", end = "\r")
        except:
            pass
        return
    print(string, end=end)


class Helper:
    # Start Ganache client with connection to infura
    # Create web3 instance
    # Recursive function used to first get the latest block and then
    # ...fork the chain latest possible
    def initiate_rpc(self, 
                         NUMBER_OF_GOOD_CONTRIBUTORS, 
                         NUMBER_OF_BAD_CONTRIBUTORS, 
                         NUMBER_OF_FREERIDER_CONTRIBUTORS, 
                         NUMBER_OF_INACTIVE_CONTRIBUTORS,
                         MINIMUM_ROUNDS,
                         pytorch_model,
                         latestBlock=1000000, 
                         infura_url=None, 
                         manual_setup=False,
                         fork=True,
                         accounts=None):
        global w3
        NUMBER_OF_CONTRIBUTORS = NUMBER_OF_GOOD_CONTRIBUTORS \
                                    + NUMBER_OF_BAD_CONTRIBUTORS \
                                    + NUMBER_OF_FREERIDER_CONTRIBUTORS \
                                    + NUMBER_OF_INACTIVE_CONTRIBUTORS
        if not infura_url:
            with open("infuraurl", "r") as inf:
                infura_url = inf.read().strip()
        if fork:
            if not manual_setup:
                port = 8545
                process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
                stdout, stderr = process.communicate()
                for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
                    data = [x for x in process.split(" ") if x != '']
                    if (len(data) <= 1):
                        continue

                    os.kill(int(data[1]), signal.SIGKILL)
                command = "ganache --fork.url='{}' -a {} -b 10".format(infura_url, NUMBER_OF_CONTRIBUTORS)
                os.system("gnome-terminal -e 'bash -c \"{}; bash\" '".format(command))
        while latestBlock == 1000000:
            time.sleep(1)
            try:
                if fork:
                    w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
                else:
                    w3 = Web3(Web3.HTTPProvider(infura_url))
                latestBlock = w3.eth.getBlock("latest").number
            except:
                latestBlock = 1000000
        
        
        #print("\n==================================================================================\n")
        print("Connected to Ethereum: {}".format(colored(w3.isConnected(), "green", attrs=['bold'])))
        print("initiated Ganache-Client @ Block Nr. {:,.0f}\n".format(latestBlock))        
        print("Total Contributers:       {}".format(NUMBER_OF_CONTRIBUTORS))
        print("Good Contributers:        {} ({:.0f}%)".format(NUMBER_OF_GOOD_CONTRIBUTORS,
                                                        NUMBER_OF_GOOD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100)) 
        print("Malicious Contributers:   {} ({:.0f}%)".format(NUMBER_OF_BAD_CONTRIBUTORS,
                                                        NUMBER_OF_BAD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Freeriding Contributers:  {} ({:.0f}%)".format(NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                        NUMBER_OF_FREERIDER_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Inactive Contributers:    {} ({:.0f}%)".format(NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                        NUMBER_OF_INACTIVE_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Learning Rounds:          {}".format(MINIMUM_ROUNDS)) 
        
        print("-----------------------------------------------------------------------------------")
        
        if fork:
            while not w3.eth.default_account:
                time.sleep(0.2)
                try:
                    w3.eth.default_account = w3.eth.accounts[0]
                except:
                    w3.eth.default_account = None
            
            if len(w3.eth.accounts) < len(self.pytorch_model.participants):
                print(rb("Nr. of Ganache Addresses <> Nr. of Model Participants"))
                print(rb(str(len(w3.eth.accounts))  + "<>" +  str(len(self.pytorch_model.participants))))
                print(rb("Increase number of unlocked accounts"))
                raise NotEnoughUnlockedAccounts()
                
        # Every user receives an address
        for ix in range(len(self.pytorch_model.participants)):
            if fork:
                self.pytorch_model.participants[ix].address = w3.toChecksumAddress(w3.eth.accounts[ix])    
            else:
                if ix == 0:
                    w3.eth.default_account = accounts[ix].address 
                self.pytorch_model.participants[ix].address = w3.toChecksumAddress(accounts[ix].address) 
                self.pytorch_model.participants[ix].privateKey = accounts[ix].privateKey           
                
            
        for i, acc in enumerate(self.pytorch_model.participants):
            if acc.futureAttitude == "good":
                prefix = "FAIR"
            elif acc.futureAttitude == "freerider":
                prefix = "FREE"
            elif acc.futureAttitude == "inactive":
                prefix = "AFK "
            else:
                prefix = "MAL."
            bal = w3.eth.get_balance(acc.address)
            print("{:<17} {} with {:<4,.1f} ETH | {} USER".format("Account initiated", 
                                                           "@ Address "+acc.address[0:25]+"...",
                                                           bal/1e18,
                                                           prefix))
        print("-----------------------------------------------------------------------------------")
        self.w3 = w3
        return w3, latestBlock

    
    
    def initialize(self):
        with open("build/abi.txt") as abiFile:
            abi = re.sub("\n|\t|\ ", "", abiFile.read())
        with open("build/bytecode.txt") as abiFile:
            bytecode = abiFile.read().strip()
        return self.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    
    
    def initializeModel(self):
        with open("build/abi_model.txt") as abiFile:
            abi = re.sub("\n|\t|\ ", "", abiFile.read())
        with open("build/bytecode_model.txt") as abiFile:
            bytecode = abiFile.read().strip()
        return self.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    
    
    def buildTx(self, _from, _to, _value=0):
        assert(_to != "0x0000000000000000000000000000000000000000")
        _from = w3.toChecksumAddress(_from)
        _to = w3.toChecksumAddress(_to)        
        return {
            'from': _from,
            'to': _to,
            'value': _value
            #,'gas': 300000,
            #'maxFeePerGas': self.w3.toWei(250, 'gwei'),
            #'maxPriorityFeePerGas': self.w3.toWei(5, 'gwei'),
        }
    
    
    
    def buildNonForkTx(self, addr, nonce, to=None, value=0, data=None):
        if data:
            return {'chainId': 3,
                    'from': addr,
                    'to': to,
                    'gas': 10000000,
                    'maxFeePerGas': w3.toWei('12', 'gwei'),
                    'maxPriorityFeePerGas': w3.toWei('2', 'gwei'),
                    'nonce': nonce,
                    'value': value,
                    'data': data}
        if to:
            return {'chainId': 3,
                    'from': addr,
                    'to': to,
                    'gas': 10000000,
                    'maxFeePerGas': w3.toWei('12', 'gwei'),
                    'maxPriorityFeePerGas': w3.toWei('2', 'gwei'),
                    'nonce':nonce,
                    'value': value}
        else:
            return {'chainId': 3,
                    'from': addr,
                    'gas': 10000000,
                    'maxFeePerGas': w3.toWei('12', 'gwei'),
                    'maxPriorityFeePerGas': w3.toWei('2', 'gwei'),
                    'nonce':nonce,
                    'value': value}
    
    
    def bar(self, i, l):
        p = "-" * (i+1)
        r = "." *((l-1)-i)
        _print("{}{}".format(p, r), end="\r")
        
    
    
class FLManager(Helper):
    
    def __init__(self, pytorch_model, manual_ganache_setup=False):
        self.w3 = None
        self.latestBlock = None
        self.manager = None
        self.challenge_contract = None
        self.pytorch_model = pytorch_model
        self.modelOf = {}
        self.manual_setup = manual_ganache_setup
        
        self.gas_deploy = []
        self.txHashes   = []
        
    
    def init(self, 
             NUMBER_OF_GOOD_CONTRIBUTORS, 
             NUMBER_OF_BAD_CONTRIBUTORS, 
             NUMBER_OF_FREERIDER_CONTRIBUTORS, NUMBER_OF_INACTIVE_CONTRIBUTORS, 
             MINIMUM_ROUNDS, 
             infuraurl=None, 
             fork=True,
             accounts=None): 
        
        self.fork = fork
        self.w3, self.latestBlock = super().initiate_rpc(NUMBER_OF_GOOD_CONTRIBUTORS=NUMBER_OF_GOOD_CONTRIBUTORS,
                                                         NUMBER_OF_BAD_CONTRIBUTORS=NUMBER_OF_BAD_CONTRIBUTORS,
                                                         NUMBER_OF_FREERIDER_CONTRIBUTORS=NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                         NUMBER_OF_INACTIVE_CONTRIBUTORS=NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                         MINIMUM_ROUNDS=MINIMUM_ROUNDS, pytorch_model=self.pytorch_model,
                                                         infura_url=infuraurl, manual_setup=self.manual_setup, fork=fork,
                                                         accounts=accounts)
        self.manager = super().initialize()
        return self
    
    
    # Deploy contract and initiate proxy
    def buildContract(self):
        if self.fork:
            genesisHash = self.manager.constructor().transact()  # Build Contract
        else:
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account) 
            depl = super().buildNonForkTx(self.w3.eth.default_account, nonce)   
            depl = self.manager.constructor().buildTransaction(depl)
            signed = self.w3.eth.account.signTransaction(depl, private_key=self.pytorch_model.participants[0].privateKey)

            genesisHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
        receipt = self.w3.eth.waitForTransactionReceipt(genesisHash,
                                                        timeout=600, 
                                                        poll_latency=1)
        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex()))
        
        self.manager.address = receipt.contractAddress
        print("\n{:<17} {} | {}\n".format("Manager deployed", 
                                          "@ Address " + self.manager.address, 
                                          genesisHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        return 
    
    
    
    def getModelOf(self, p, c):
        return self.manager.functions.ModelOf(p.address, c).call({"to": self.manager.address,
                                                                  "from": p.address})
    
    
    def getModelCountOf(self, p):
        return self.manager.functions.ModelCountOf(p.address).call({"to": self.manager.address,
                                                                  "from": p.address})
    
    
    def deployChallengeContract(self, *args):
        print(b("Starting simulation..."))
        print(b("-----------------------------------------------------------------------------------"))
        min_buyin, max_buyin, reward, min_rounds, punishment, freerider_fee = args
        p1_collateral = self.pytorch_model.participants[0].collateral
        value = reward + p1_collateral
        deployer =  self.pytorch_model.participants[0].address
        modelHash = self.pytorch_model.participants[0].modelHash
        if self.fork:
            tx = super().buildTx(deployer, self.manager.address, value)
            txHash = self.manager.functions.deployModel(modelHash,
                                                        min_buyin, 
                                                        max_buyin, 
                                                        reward,
                                                        min_rounds,
                                                        punishment,
                                                        freerider_fee).transact(tx)
        else:          
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address) 
            depl = super().buildNonForkTx(deployer, nonce, self.manager.address, value)   
            depl = self.manager.functions.deployModel(modelHash,
                                                      min_buyin, 
                                                      max_buyin, 
                                                      reward,
                                                      min_rounds,
                                                      punishment,
                                                      freerider_fee).buildTransaction(depl)
            signed = self.w3.eth.account.signTransaction(depl, private_key=self.pytorch_model.participants[0].privateKey)
            txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
            
        receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                        timeout=600, 
                                                        poll_latency=1)

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildChallenge", receipt["transactionHash"].hex()))
        self.challenge_contract = super().initializeModel()
        c = self.getModelCountOf(self.pytorch_model.participants[0])
        self.challenge_contract.address = self.getModelOf(self.pytorch_model.participants[0], c)
        print("\n{:<17} {} | {}\n".format("Model deployed", 
                                          "@ Address " + self.challenge_contract.address, 
                                          txHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                                                           self.pytorch_model.participants[0].address[0:16] + "...", 
                                                           txHash.hex()[0:6] + "...", 
                                                           p1_collateral
                                                           ))

        self.pytorch_model.participants[0].isRegistered = True
        return (self.challenge_contract, self.challenge_contract.address) + args
    
   

class FLChallenge(FLManager):
    def __init__(self, manager, configs, pyTorchModel):
        self.manager = manager
        self.w3 = manager.w3
        self.model, self.modelAddress = configs[:2]
        self.pytorch_model = pyTorchModel
        self.MIN_BUY_IN, self.MAX_BUY_IN , self.REWARD, self.MIN_ROUNDS, = configs[2:-2]
        self.PUNISHMENT_FACTOR = configs[-2]
        self.FREERIDER_FACTOR  = configs[-1]
        self.fork = manager.fork
        
        self.gas_feedback = [] 
        self.gas_register = [] 
        self.gas_slot     = [] 
        self.gas_weights  = [] 
        self.gas_close    = [] 
        self.gas_deploy   = [] 
        self.gas_exit     = []
        self.txHashes     = []
        
        self._reward_balance = [self.REWARD]
        self._punishments = []

              
        
    def registerAllUsers(self):
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.isRegistered:
                continue
            if self.fork:
                tx = super().buildTx(acc.address, self.modelAddress, acc.collateral)
                txHash = self.model.functions.register().transact(tx)
            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                reg = super().buildNonForkTx(acc.address, nonce, self.modelAddress, acc.collateral)   
                reg = self.model.functions.register().buildTransaction(reg)
                signed = self.w3.eth.account.signTransaction(reg, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            txs.append(txHash)
            bal = self.w3.eth.get_balance(w3.eth.default_account)
            acc.isRegistered = True
            print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                                                           acc.address[0:16] + "...", 
                                                           txHash.hex()[0:6] + "...", 
                                                           acc.collateral
                                                           ))
        
        l = len(txs)
        for i, txHash in enumerate(txs):
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_register.append(receipt["gasUsed"])
            self.txHashes.append(("register",receipt["transactionHash"].hex()))
        _print("-----------------------------------------------------------------------------------", "\n")
        
    
    def getHashedWeightsOf(self, user):
        return self.model.functions.weightsOf(user.address,self.pytorch_model.round-1).call({"to": self.modelAddress})
    
    
    def getGlobalReputationOfUser(self, user):
        return self.model.functions.GlobalReputationOf(user).call({"to": self.modelAddress})
        
    
    def getRoundReputationOfUser(self, user):
        return self.model.functions.RoundReputationOf(user).call({"to": self.modelAddress})
    
    
    def getRewardLeft(self):
        return self.model.functions.rewardLeft().call({"to": self.modelAddress})

    
    def usersProvideHashedWeights(self):
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Account inactive:", 
                                                                         acc.address[0:16] + "...", 
                                                                         "   ...   ",
                                                                         self.getGlobalReputationOfUser(acc.address)
                                                                         ))
                continue
            if self.fork:
                tx = super().buildTx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.provideHashedWeights(acc.hashedModel, acc.secret).transact(tx)

            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                hw = super().buildNonForkTx(acc.address, nonce, self.modelAddress, 0)   
                hw =  self.model.functions.provideHashedWeights(acc.hashedModel, acc.secret).buildTransaction(hw)
                signed = self.w3.eth.account.signTransaction(hw, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Weights provided:", 
                                                                         acc.address[0:16] + "...", 
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.getGlobalReputationOfUser(acc.address)
                                                                         ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_weights.append(receipt["gasUsed"])
            self.txHashes.append(("weights", receipt["transactionHash"].hex()))
        _print("-----------------------------------------------------------------------------------\n")
        

             
    def giveFeedback(self, feedbackGiver, target, score):
        time.sleep(0.1)
        tx = super().buildTx(feedbackGiver.address, self.modelAddress, 0)
        #data = "0x" + encode_abi(['address', 'uint'], [target, score]).hex()
        if target in feedbackGiver.cheater:
            score = -1
        try:
            if self.fork:
                txHash = self.model.functions.feedback(target.address, score).transact(tx)
            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                fe = super().buildNonForkTx(acc.address, nonce, self.modelAddress, 0)   
                fe =  self.model.functions.feedback(target.address, score).buildTransaction(fe)
                signed = self.w3.eth.account.signTransaction(fe, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
        except ContractLogicError as e:
            if "FRC" in str(e):
                input("Inactive users found - such users do not provide hashed weights.. \nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [WAIT_DELAY])
                time.sleep(1)
                txHash = self.model.functions.feedback(target.address, score).transact(tx)
            else:
                print(rb("Encountered error at feedback function"))
                raise 
                
        assert(txHash != None)
        
        if score == 1:
            target.roundRep += 1 * self.getGlobalReputationOfUser(feedbackGiver.address)
            rep = "Positive"
            pre = "+"
            col = "green"

        elif score == 0:
            rep = "Neutral"
            pre = "+"
            col = None
        else:
            target.roundRep -= 1 * self.getGlobalReputationOfUser(feedbackGiver.address)
            rep = "Negative"
            pre = "-"
            col = "red"
        fb = "Feedback:".format(rep)
        
        print(colored("{:<11} {}   |" \
            " {}  | {}{:>25,.0f} WEI".format(fb, 
                                    feedbackGiver.address[0:7]+"... --> "+target.address[0:7]+"...", 
                                    txHash.hex()[0:6] + "...",
                                    pre,
                                    self.getGlobalReputationOfUser(feedbackGiver.address)), col))
        return txHash
        
            
    
    def returnStats(self):
        print("\n==================================================================================\n")
        print("\n{:<8}{:^32}  {:^32}".format(f"ROUND {self.pytorch_model.round}","GLOBAL REPUTATION", "ROUND REPUTATION"))
        for acc in self.pytorch_model.participants:
            gs = self.getGlobalReputationOfUser(acc.address)
            rs = self.getRoundReputationOfUser(acc.address)
            print("{}..: {:>27,.0f}  {:>27,.0f} WEI".format(acc.address[0:7],gs,rs))
        print("\n==================================================================================\n")
    
            
    def feedbackRound(self, fbm):
        txs = []
        for user in self.pytorch_model.participants:
            user_votes = fbm[user.id]
            for ix, vote in enumerate(user_votes):
                if user.id == ix:
                    continue
                if user.attitude == "inactive":
                    continue
                txHash = self.giveFeedback(user, self.pytorch_model.participants[ix], int(vote))
                txs.append(txHash)
           
        l = len(txs)
        for i, txHash in enumerate(txs):
            if txHash == None:
                continue
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_feedback.append(receipt["gasUsed"])
            self.txHashes.append(("feedback", receipt["transactionHash"].hex()))
        for user in self.pytorch_model.participants:
            user._roundrep.append(self.getRoundReputationOfUser(user.address))
            
        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.getRoundReputationOfUser(user.address))
        _print("                                                   ")
        print("\n-----------------------------------------------------------------------------------")
        
    def buildFeedbackBytes(self, a, v):
        fbb = ""
        for i in range(len(a)):
            fbb += encode_single('address', a[i]).hex()[24:]

        for i in range(len(v)):
            fbb += encode_single('int256', v[i]).hex()

        return fbb

                
    
    def quickFeedbackRound(self, fbm):
        print("Users exchanging feedback...")
        txs = []
        for user in self.pytorch_model.participants:
            addrs = []
            votes = []
            user_votes = fbm[user.id]
            for ix, vote in enumerate(user_votes):
                if user.id == ix:
                    continue
                if user.attitude == "inactive":
                    continue
                if ix in [i.id for i in self.pytorch_model.disqualified]:
                    continue
                votee = [_u for _u in self.pytorch_model.participants if _u.id == ix][0]
                addrs.append(votee.address)
                votes.append(int(vote))
                votee.roundRep = votee.roundRep + self.getGlobalReputationOfUser(user.address) * int(vote)
            
            fbb = self.buildFeedbackBytes(addrs, votes)
            try:
                if self.fork:
                    txHash = self.w3.eth.sendTransaction({'to': self.modelAddress, 'from': user.address, 'data': fbb})
                else:          
                    nonce = self.w3.eth.get_transaction_count(user.address) 
                    hw = super().buildNonForkTx(user.address, nonce, self.modelAddress, 0, fbb)   
                    signed = self.w3.eth.account.signTransaction(hw, private_key=user.privateKey)
                    txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
                txs.append(txHash)

            except ContractLogicError as e:
                if "FRC" in str(e):
                    input("Inactive users found - such users do not "\
                              + "provide hashed weights.. \nGoing to forward time for 1 day\n")
                    
                    self.w3.provider.make_request("evm_increaseTime", [WAIT_DELAY])
                    time.sleep(1)
                    txHash = self.w3.eth.sendTransaction({'to': self.modelAddress, 
                                                          'from': user.address, 
                                                          'data': fbb, 
                                                          "gas":500000})
                    txs.append(txHash)
                else:
                    print(rb("Encountered error at feedback function"))
                    raise 

           
        l = len(txs)
        for i, txHash in enumerate(txs):
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_feedback.append(receipt["gasUsed"])
            self.txHashes.append(("feedback", receipt["transactionHash"].hex()))

        
        for user in self.pytorch_model.participants:
            user._roundrep.append(self.getRoundReputationOfUser(user.address))
            
        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.getRoundReputationOfUser(user.address))
            
        _print("                                                   ")
        print("\n-----------------------------------------------------------------------------------")
        
        
    
    def closeRound(self):
        if "inactive" in [acc.attitude for acc in self.pytorch_model.participants]:
                input("Inactive users found - such users do not provide feedback.. " \
                          + "\nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [WAIT_DELAY])
        
        print(b(f"\nSettle round: {self.pytorch_model.round}"))
                
        if self.fork:
            tx = super().buildTx(self.w3.eth.default_account, self.modelAddress, 0)
            txHash = self.model.functions.closeRound().transact(tx)
            
        else:          
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address) 
            cl = super().buildNonForkTx(self.pytorch_model.participants[0].address, 
                                        nonce, 
                                        self.modelAddress, 
                                        0)   
            cl =  self.model.functions.closeRound().buildTransaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = self.w3.eth.account.signTransaction(cl, private_key=pk)
            txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
        receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)          

        self.txHashes.append(("close", receipt["transactionHash"].hex()))
        self.gas_close.append(receipt["gasUsed"])
        assert(len(receipt.logs) > 0)
        self.pytorch_model.round += 1
        self._reward_balance.append(self.getRewardLeft())
        _print("\n-----------------------------------------------------------------------------------\n")
        return receipt
    
    
    
    def userRegisterSlot(self):
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Account inactive:", 
                                                                         acc.address[0:16] + "...", 
                                                                         "   ...   ",
                                                                         self.getGlobalReputationOfUser(acc.address)
                                                                         ))
                continue
           
            reservation = Web3.solidityKeccak(['bytes32', 'uint256'], 
                                              [acc.hashedModel, 
                                               acc.secret]).hex()
            if self.fork:
                tx = super().buildTx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.registerSlot(reservation).transact(tx)
            else:          
                nonce = w3.eth.get_transaction_count(acc.address) 
                sl = super().buildNonForkTx(acc.address, nonce, self.modelAddress, 0)   
                sl =  self.model.functions.registerSlot(reservation).buildTransaction(sl)
                signed = w3.eth.account.signTransaction(sl, private_key=acc.privateKey)
                txHash = w3.eth.sendRawTransaction(signed.rawTransaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Slot registered: ", 
                                                                         acc.address[0:16] + "...", 
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.getGlobalReputationOfUser(acc.address)
                                                                         ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_slot.append(receipt["gasUsed"])
            self.txHashes.append(("slot", receipt["transactionHash"].hex()))
        _print("-----------------------------------------------------------------------------------\n")
        return 
    
    
    
    def exitSystem(self):
      
        print(b(f"Terminating Model..."))
       
        txs = []
        for acc in self.pytorch_model.participants:
            
            if self.fork:
                tx = super().buildTx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.exitModel().transact(tx)
            else:          
                nonce = w3.eth.get_transaction_count(acc.address) 
                ex = super().buildNonForkTx(acc.address, nonce, self.modelAddress, 0)   
                ex =  self.model.functions.exitModel().buildTransaction(ex)
                signed = w3.eth.account.signTransaction(ex, private_key=acc.privateKey)
                txHash = w3.eth.sendRawTransaction(signed.rawTransaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>27,.0f} WEI".format("Account exited:  ", 
                                                             acc.address[0:16] + "...", 
                                                             txHash.hex()[0:6] + "...",
                                                             self.w3.eth.get_balance(acc.address)
                                                             ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            self.bar(i, l)
            receipt = self.w3.eth.waitForTransactionReceipt(txHash, 
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_exit.append(receipt["gasUsed"])
            self.txHashes.append(("exit", receipt["transactionHash"].hex()))
        _print("-----------------------------------------------------------------------------------\n")
    

    
    def print_round_summary(self, receipt):
        punishEvent   = self.model.events.Punishment()
        rewardEvent   = self.model.events.Reward()
        endRoundEvent = self.model.events.EndRound()
        disqualiEvent = self.model.events.Disqualification()
        
        end = endRoundEvent.processReceipt(receipt)
        rew = rewardEvent.processReceipt(receipt)
        pun = punishEvent.processReceipt(receipt)
        dis = disqualiEvent.processReceipt(receipt)
        
        if len(end) > 0:
            for ev in end:
                print(b("\nEND OF ROUND {}".format(ev.args.round+1)))
                print(b("VALID VOTES:      {}".format(ev.args.validVotes)))
                print(b("REWARD PER VOTE:  {:,.0f}".format(ev.args.rewardPerVote)))
                print(b("TOTAL PUNISHMENT: {:,.0f}\n".format(ev.args.totalPunishment)))
            print("-----------------------------------------------------------------------------------\n")
        if len(rew) > 0:
            print(b("REWARDED USERS"))
            for ev in rew:
                if ev.args.roundScore > 0:
                    print(green("USER @ {}".format(ev.args.user)))
                    print(green("ROUND SCORE:      {:,.0f}".format(ev.args.roundScore)))
                    print(green("TOTAL REWARD:     {:,.0f}".format(ev.args.win)))
                    print(green("NEW REPUTATION:   {:,.0f}\n".format(ev.args.newReputation)))
            print("-----------------------------------------------------------------------------------\n")
        if len(pun) > 0:
            print(b("PUNISHED USERS"))
            for ev in pun:
                self._punishments.append((self.pytorch_model.round-1, ev.args.loss))
                print(red("USER @ {}".format(ev.args.victim)))
                print(red("ROUND SCORE:      {:,.0f}".format(ev.args.roundScore)))
                print(red("TOTAL LOSS:       {:,.0f}".format(ev.args.loss)))
                print(red("NEW REPUTATION:   {:,.0f}\n".format(ev.args.newReputation)))
            print("-----------------------------------------------------------------------------------\n")
        if len(dis) > 0:
            print(b("DISQUALIFIED USERS"))
            for ev in dis:
                self._punishments.append((self.pytorch_model.round-1, ev.args.loss))
                for user in self.pytorch_model.participants:
                    if ev.args.victim == user.address:
                        user.disqualified = True
                        self.pytorch_model.disqualified.append(user)
                        self.pytorch_model.participants.remove(user)
                print(red("USER @ {}".format(ev.args.victim)))
                print(red("ROUND SCORE:      {:,.0f}".format(ev.args.roundScore)))
                print(red("TOTAL LOSS:       {:,.0f}".format(ev.args.loss)))
                print(red("NEW REPUTATION:   {:,.0f}\n".format(ev.args.newReputation)))
            print("-----------------------------------------------------------------------------------\n")
        print()
    

        
    def simulate(self, rounds):
        hashedWeights = []
        self.registerAllUsers()
        
        for i in range(rounds):
            print(b(f"Round {self.pytorch_model.round} starts..."))
            self.pytorch_model.update_users_attitude()

            self.pytorch_model.federated_training()

            self.pytorch_model.let_malicious_users_do_their_work()
            self.pytorch_model.let_freerider_users_do_their_work()
            
            self.userRegisterSlot()

            self.usersProvideHashedWeights()

            self.pytorch_model.exchange_models()
            
            self.pytorch_model.verify_models({u.id: self.getHashedWeightsOf(u) for u in self.pytorch_model.participants})

            feedback = self.pytorch_model.evaluation()
            
            self.quickFeedbackRound(feedback)

            receipt = self.closeRound()

            self.pytorch_model.the_merge([user for user in self.pytorch_model.participants if user.roundRep > 0])
                   
            self.print_round_summary(receipt)

            print(b(f"Round {self.pytorch_model.round-1} completed:"))
            
            for user in self.pytorch_model.participants+self.pytorch_model.disqualified:
                user._globalrep.append(self.getGlobalReputationOfUser(user.address))
                i, j = user._globalrep[-2:]
                print(b("{}  {:>25,.0f} -> {:>25,.0f}".format(user.address[0:16]+"...",i,j)))
            
            print(b("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n"))
            
        self.exitSystem()
            
            
    
    def visualize_simulation(self):
        
        accuracy = [0] + self.pytorch_model.accuracy
        loss = [self.pytorch_model.loss[0]] + self.pytorch_model.loss

        f, axs = plt.subplots(1, 4,figsize=(16, 3),  gridspec_kw={'width_ratios': [0.8,2,2,2],
                                                                      'height_ratios': [1]})
        colors = ["#00629b", "#629b00", "#000000", "#d93e6a"]

        participants =self.pytorch_model.participants + self.pytorch_model.disqualified

        x = list(range(0,len(accuracy)))
        #x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]

        y = accuracy
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        axs[1].plot(x, y, color=colors[0], linewidth=2.5) 
        twin = axs[1].twinx()
        y = loss
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        twin.plot(x, y, color=colors[1], linewidth=2.5) 



        x = list(range(len(participants[0]._globalrep)))
        x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]


        # plotting the points  
        yy=[]
        for i, user in enumerate(participants):
            y = [item for sublist in zip(user._globalrep, user._globalrep) for item in sublist]
            axs[2].plot(x, y, linewidth=2.5, color=user.color) 



        pun = {}
        for i, j in self._punishments:
            if i in pun.keys():
                pun[i] += j
            else:
                pun[i] = j

        rew = list()
        for i, j in enumerate(self._reward_balance):
            if i in pun.keys():
                rew.append(j+pun[i])
            else:
                rew.append(j)    

        y_reward = [item for sublist in zip(self._reward_balance,self._reward_balance) for item in sublist]
        y2_reward = [item for sublist in zip(rew,rew) for item in sublist]
        x_reward = list(range(len(self._reward_balance)))
        x_reward = [item for sublist in zip(x_reward,(np.array(x_reward)+1).tolist()) for item in sublist]


        axs[3].plot(x_reward,y_reward, label="reward", color=colors[0], linewidth=2.5)
        axs[3].plot(x_reward,y2_reward, label="reward +\npunishments", color=colors[1], linewidth=2.5)
        axs[3].fill_between(x_reward,y_reward, y2_reward, alpha=0.2, hatch=r"//", color = colors[1])


        axs[0].text(0, 0.1, f'dataset={self.pytorch_model.DATASET}\n'\
                                 + f'epoches={self.pytorch_model.EPOCHES}\n' \
                                 + f'rounds={self.pytorch_model.round-1}\n' \
                                 + f'users={self.pytorch_model.NUMBER_OF_CONTRIBUTERS}\n' \
                                 + f'malicious={self.pytorch_model.NUMBER_OF_BAD_CONTRIBUTORS}\n'\
                                 + f'copycat={self.pytorch_model.NUMBER_OF_FREERIDER_CONTRIBUTORS}', fontsize=15)
        axs[0].set_axis_off()
        
        axs[1].set_xlabel('rounds\n(a)', fontsize=14)
        axs[2].set_xlabel('rounds\n(b)', fontsize=14)
        axs[3].set_xlabel('rounds\n(c)', fontsize=14)
        #axs[0].set_ylabel(f'users={participants};\n malicious={malicious_users};\n copycat={sneaky_freerider}', fontsize=14)
        axs[1].set_ylabel('Avg. Accuracy', fontsize=14)
        twin.set_ylabel('Avg. Loss', fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        axs[2].set_ylabel('GRS', fontsize=14)
        axs[3].set_ylabel('Contract Balance', fontsize=14)

        axs[2].tick_params(axis='both', which='major', labelsize=14)
        axs[3].tick_params(axis='both', which='major', labelsize=14)
        
        if len(x) > 20:
            axs[1].set_xticks([i for i in x if i%5==0 or i == 0])
            axs[2].set_xticks([i for i in x if i%5==0 or i == 0])
            axs[3].set_xticks([i for i in x if i%5==0 or i == 0])
        else:
            axs[1].set_xticks([i for i in x])
            axs[2].set_xticks([i for i in x])
            axs[3].set_xticks([i for i in x])
    
        axs[1].set_xlim(0,max(x))
        
        axs[2].yaxis.get_offset_text().set_fontsize(14)
        axs[3].yaxis.get_offset_text().set_fontsize(14)
        
        axs[1].grid()
        axs[2].grid()
        axs[3].grid()

        lgnd = axs[3].legend(fontsize=10)

        # giving a title to my graph 
        #axs[1].set_title(f'users={participants}; malicious={malicious_users}; copycat={sneaky_freerider}', fontsize=10) 

        # function to show the plot 
        plt.tight_layout(pad=1)
        plt.savefig(f"./pictures/{self.pytorch_model.DATASET}_simulation.pdf", bbox_inches='tight')
        plt.show() 