// SPDX-License-Identifier: Apache-2.0
pragma solidity =0.8.9;

import "./math/PRBMathSD59x18.sol";
import "./math/PRBMathUD60x18.sol";

contract PubFLModel {
    using PRBMathSD59x18 for int256;
    using PRBMathUD60x18 for uint256;

    bytes32 public modelHash;
    
    uint8 public round = 0;
    uint8 public votesPerRound;
    uint8 public punishfactor;
    uint8 public min_rounds;
    
    uint public nrOfParticipants;
    uint public nrOfProvidedHashedWeights;
    uint public initTS;
    uint public min_collateral;
    uint public max_collateral;
    uint public totalReward;
    uint public rewardPerRound;
    uint public rewardLeft;
    uint public roundStart;
    uint public freeriderPenalty;
    uint public ONE_DAY = 864e2;

    address[] public participants;
    address[] punishedAddresses;

    mapping(address => bool) public isRegistered;
    mapping(address => uint) public GlobalReputationOf;
    mapping(address => int) public RoundReputationOf;
    mapping(address => uint8) public nrOfVotesOfUser;
    mapping(address => bool) public whitelistedForRewards;
    mapping(address => uint8) roundOfUser;
    mapping(address => mapping(address => bool)) public hasVoted;
    mapping(address => mapping(address => bool)) public votedPositiveFor;
    mapping(address => mapping(uint8 =>bytes32)) public secretOf;
    mapping(address => mapping(uint8 => bytes32)) public weightsOf;

    modifier onlyRegisteredUsers {
        require(isRegistered[msg.sender], "SNR");
        _;
    }

    modifier feedbackRoundOpened {
        require(nrOfProvidedHashedWeights == nrOfParticipants || roundStart + ONE_DAY < block.timestamp, "FRC");
        _;
    }

    modifier feedbackRoundClosed{
        require(nrOfProvidedHashedWeights != nrOfParticipants && roundStart + ONE_DAY > block.timestamp, "NA");
        require(weightsOf[msg.sender][round] == bytes32(0), "WFE");
        _;
    }

    modifier onlyValidTargets(address target) {
        require(target != msg.sender, "SET");
        require(!hasVoted[msg.sender][target], "VAC");
        _;
    }

    modifier onlyNotYetRegisteredUsers {
        require(!isRegistered[msg.sender], "SAR");
        _;
    }

    modifier hasNotYetProvidedWeights {
        require(weightsOf[msg.sender][round] == bytes32(0), "SAP");
        _;
    }  

    event FederatedLerningModelDeployed(uint initTS, 
                                        uint max_collateral, 
                                        uint min_collateral, 
                                        uint total_reward, 
                                        uint8 min_rounds,
                                        uint freerider_fee);

    event Registered(address user, 
                     uint reputationValue,
                     uint totalCollateral, 
                     uint numberOfContributers);

    event Feedback(address target, 
                   address user,
                   uint globalReputation, 
                   int newRoundReputation);

    event EndRound(uint8 round, 
                   uint8 validVotes,
                   uint rewardPerVote, 
                   uint totalPunishment);

    event Punishment(address victim,
                     int roundScore,
                     uint loss,
                     uint newReputation);

    event PassivPunishment(address victim,
                           int roundScore,
                           uint loss,
                           uint newReputation);

    event Disqualification(address victim,
                           int roundScore,
                           uint loss,
                           uint newReputation);

    event Reward(address user,
                 int roundScore,
                 uint win,
                 uint newReputation);


    constructor(bytes32 _modelHash, uint _min_collateral, uint _max_collateral, uint _reward, 
                uint8 _min_rounds, uint8 _punishfactor, uint8 _freeriderPenalty) payable {

        // Initialize Contract
        initTS      = block.timestamp;
        roundStart  = block.timestamp;
        modelHash   = _modelHash;
        min_collateral = _min_collateral;
        max_collateral = _max_collateral;
        totalReward = _reward ;
        min_rounds  = _min_rounds;
        punishfactor =_punishfactor;
        freeriderPenalty = (min_collateral * _freeriderPenalty) / 100;
        rewardPerRound = totalReward / min_rounds;
        rewardLeft = totalReward;

        emit FederatedLerningModelDeployed(initTS, 
                                           min_collateral, 
                                           max_collateral, 
                                           totalReward, 
                                           min_rounds, 
                                           freeriderPenalty);
    }

    // Register participants
    function register() public payable onlyNotYetRegisteredUsers {
        require(msg.value >= min_collateral && msg.value <= max_collateral, "NWR");
        registrationProcess(msg.sender);
    }

    // Register initiator of model
    function register(address initiator) public payable onlyNotYetRegisteredUsers {
        require(msg.value >= min_collateral && msg.value <= max_collateral, "NWR");
        // Require Staking here
        registrationProcess(initiator);
    }

    // Registration helper
    function registrationProcess(address user) internal {
        isRegistered[user] = true;
    	GlobalReputationOf[user] = msg.value;
        nrOfParticipants += 1;
        participants.push(user);
        roundOfUser[user] = 1;
        emit Registered(user, msg.value, address(this).balance, nrOfParticipants);
    }

    // Register Slot
    function registerSlot(bytes32 _secret) public onlyRegisteredUsers hasNotYetProvidedWeights  {
        secretOf[msg.sender][round] = _secret;
    }

    // Timestamp weights to the chain
    function provideHashedWeights(bytes32 hashedWeights, uint salt) public onlyRegisteredUsers hasNotYetProvidedWeights {
        require(secretOf[msg.sender][round] == keccak256(abi.encodePacked(hashedWeights, salt)), "NKS");
        weightsOf[msg.sender][round] = hashedWeights;
        nrOfProvidedHashedWeights += 1;
    }


    function feedback(address target, int score) public onlyRegisteredUsers onlyValidTargets(target) feedbackRoundOpened {
        //(address target, int score) = abi.decode(data, (address, int));
        hasVoted[msg.sender][target]=true;
        nrOfVotesOfUser[msg.sender] += 1;
        votesPerRound += 1;
        if (score == 1) {
            votedPositiveFor[msg.sender][target]=true;
            RoundReputationOf[target] += 1 * int(GlobalReputationOf[msg.sender]);
        }
        if (score == -1) {
            votedPositiveFor[msg.sender][target]=false;
            RoundReputationOf[target] -= 1  * int(GlobalReputationOf[msg.sender]); 
        }
        if (score == 0) {
            votedPositiveFor[msg.sender][target]=false;
        }
        emit Feedback(target, msg.sender, GlobalReputationOf[msg.sender], RoundReputationOf[target]);  
    }

    // Close feedback round
    // @notice The round closes either if every registered participant provided feedback to everyone else 
    // or if 1 day (86400 seconds) passed since model initialization
    function closeRound() public returns(bool roundClosed){
        uint votes;
        for (uint i=0; i<participants.length; i++) { 
            if (nrOfVotesOfUser[participants[i]] < nrOfParticipants - 1) {
                if (block.timestamp > roundStart + ONE_DAY * 1) {
                    settle();
                    return true;
                }
            } else {
                votes += 1; 
            }
        } if (votes == nrOfParticipants){
            settle();
            return true;
        }
        return false;
    }

    function settle() internal {
        uint totalPunishment;
        uint freeriderLock;       

        // First round users pay their anti-freerider fee
        for (uint i=0; i<participants.length; i++) { 
            if (roundOfUser[participants[i]] == 1) {
                GlobalReputationOf[participants[i]] = GlobalReputationOf[participants[i]] - freeriderPenalty;
                freeriderLock += freeriderPenalty;
            }
        }

        // Punish malicious users
        for (uint i=0; i<participants.length; i++) { 
            if (isRegistered[participants[i]]) {
                if (RoundReputationOf[participants[i]] < 0) {
                    votesPerRound -= nrOfVotesOfUser[participants[i]];
                    if (GlobalReputationOf[participants[i]] > min_collateral / punishfactor) {
                        punishedAddresses.push(participants[i]);
                        whitelistedForRewards[participants[i]] = false; 
                        uint punishment = uint(GlobalReputationOf[participants[i]] / punishfactor);
                        GlobalReputationOf[participants[i]] = GlobalReputationOf[participants[i]] - punishment;
                        totalPunishment += punishment;
                        emit Punishment(participants[i], 
                                        RoundReputationOf[participants[i]], 
                                        punishment, 
                                        GlobalReputationOf[participants[i]]);
                    } else {
                        isRegistered[participants[i]] = false;
                        punishedAddresses.push(participants[i]);
                        whitelistedForRewards[participants[i]] = false;
                        totalPunishment += GlobalReputationOf[participants[i]];
                        emit Disqualification(participants[i], 
                                            RoundReputationOf[participants[i]], 
                                            GlobalReputationOf[participants[i]], 
                                            0);
                        GlobalReputationOf[participants[i]] = 0;
                        nrOfParticipants -= 1;
                        delete participants[i];
                    }
                    
                } else {
                    whitelistedForRewards[participants[i]] = true;
                }
            }
        }

        // Punish helpers of malicious users
        for (uint i=0; i<participants.length; i++) { 
            if (isRegistered[participants[i]]) {
                for (uint j=0; j < punishedAddresses.length; j++) { 
                    if (votedPositiveFor[participants[i]][punishedAddresses[j]]) {
                        votedPositiveFor[participants[i]][punishedAddresses[j]]=false;
                        votesPerRound -= nrOfVotesOfUser[participants[i]]; 
                        whitelistedForRewards[participants[i]] = false; 
                        emit PassivPunishment(participants[i], 
                                            RoundReputationOf[participants[i]], 
                                            0, 
                                            GlobalReputationOf[participants[i]]);      
                    }
                }
            }
        }

        // Pay back freerider 1st round stake to good users
        for (uint i=0; i < participants.length; i++) { 
            if (isRegistered[participants[i]]) {
                if (roundOfUser[participants[i]] == 1) {
                    if (whitelistedForRewards[participants[i]]){
                        GlobalReputationOf[participants[i]] = GlobalReputationOf[participants[i]] + freeriderPenalty;
                        freeriderLock -= freeriderPenalty;
                    } else {
                        totalPunishment += freeriderPenalty;
                        freeriderLock -= freeriderPenalty;
                    }
                }
            }
        }
       

        // Devide reward between every user who provided (non-malicious) feedback
        // Pay back freeriderLock funds to good users
        // First round users pay their anti-freerider fee
        uint rewardPerVote = 0;
        if (votesPerRound > 0 && rewardLeft >= rewardPerRound) {
            rewardLeft -= rewardPerRound;
            rewardPerVote = rewardPerRound / votesPerRound;

            for (uint i=0; i < participants.length; i++) { 
                if (isRegistered[participants[i]]) {
                    if (whitelistedForRewards[participants[i]]){
                        delete whitelistedForRewards[participants[i]];
                        uint reward = nrOfVotesOfUser[participants[i]] * rewardPerVote;
                        if (totalPunishment > 0) {
                            uint weight = uint(nrOfVotesOfUser[participants[i]]).div(uint(votesPerRound));
                            reward +=  (totalPunishment * weight) / 1e18;   
                        }    
                        GlobalReputationOf[participants[i]] += reward;
                        emit Reward(participants[i], RoundReputationOf[participants[i]], reward, GlobalReputationOf[participants[i]]);
                    }   
                }
            }
        }

        emit EndRound(round, votesPerRound, rewardPerVote, totalPunishment);

        // Reset variables
        for (uint i=0; i<participants.length; i++) { 
            if (isRegistered[participants[i]]) {
                nrOfVotesOfUser[participants[i]] = 0;     
                RoundReputationOf[participants[i]] = 0;
                roundOfUser[participants[i]] += 1;
                for (uint j=0; j<participants.length; j++) {
                    delete hasVoted[participants[i]][participants[j]];
                }
            }        
        }

        round += 1;
        votesPerRound = 0; 
        nrOfProvidedHashedWeights = 0;
        delete punishedAddresses;  
        roundStart = block.timestamp;       
    }

    // Exit contract
    function exitModel() public onlyRegisteredUsers feedbackRoundClosed {
        require(GlobalReputationOf[msg.sender] > 0, "NEF");
        uint val = GlobalReputationOf[msg.sender];
        GlobalReputationOf[msg.sender] = 0;
        for (uint i=0; i<participants.length; i++) { 
            if (participants[i] == msg.sender) {
                delete participants[i];
            }
        }
        isRegistered[msg.sender] = false;
        payable(address(msg.sender)).transfer(val);
    }

    // Fallback function parses dynamic size feedback arrays
    // @dev This allows the contract to have an arbitrary number of participants
    fallback () external {
        address[] memory ads;
        int256[] memory ints;

        assembly {
            let tmp := 0
            let tmp2 := 0

            // Skip : function selector    : 0x4 bytes
            let offset := 0x00

            // Compute the number of addresses :
            // ((array length - 0x04) - 0x20) / 0x14
            // ((array length - sizeof(function Selector)) - sizeof(uint256)) / sizeof(address)
            let adsCount := div(calldatasize(), 0x34)

            // Allocate memory for the address array
            ads := mload(0x40)
            mstore(0x40, add(ads, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ads, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)
            // shift value by 96 bits (12 bytes) to the right to keep only the relevant portion (first 20 bytes)
            // store that value at ads[i]
            // increments calldata offset by 0x14 (20 bytes)
            for {let i := 0} lt(i, adsCount) {i := add(i, 1)} {
                tmp := calldataload(offset)
                tmp := shr(96, tmp)
                mstore(add(add(ads, 0x20), mul(i, 0x20)), tmp)
                offset := add(offset, 0x14)
            }

            // Allocate memory for the address array
            ints := mload(0x40)
            mstore(0x40, add(ints, add(0x20, mul(adsCount, 0x20))))

            // Set the size of the array
            mstore(ints, adsCount)

            // Get an address from calldata on each iteration :
            // loads 0x20 bytes from calldata starting at offset : calldata[offset: offset + 0x20)      
            // store that value at ads[i]
            // increments calldata offset by 0x20 (32 bytes)
            for {let i := 0} lt(i, adsCount) {i := add(i, 1)} {
                tmp2 := calldataload(offset)
                mstore(add(add(ints, 0x20), mul(i, 0x20)), tmp2)
                offset := add(offset, 0x20)
            }
        }

        for (uint i=0; i<ads.length; i++) { 
            feedback(ads[i], ints[i]);
        }
    }
}
