import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def valueIteration(succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]], discount: float, epsilon: float = 0.001):
    '''
    Given transition probabilities and rewards, computes and returns V and
    the optimal policy pi for each state.
    - succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
    - Returns: Dictionary mapping each state to an action.
    '''
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        """
        Args:
            V : Our initial starting values for each state
            state, action : The query point at which we need to find Q(state, action)
        
        Returns:
            Q(state, action)
        """
        # Return Q(state, action) based on V(state)
        # Q_val = sum{T[s,a,s'](R[s,a,s'] + discount*V[s'])}
        # T, R, discount given by succAndRewardProb
        Q_val = 0
        for outcome in succAndRewardProb[(state, action)]:
            next_state, prob, reward = outcome
            Q_val += prob*(reward + discount*V[next_state])
        return Q_val
        

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        """
        Args:
            V : Value at each state
        
        Returns:
            Optimum policy for that state (best action)
        """
        # Return the policy given V.
        # Remember the policy for a state is the action that gives the greatest Q-value.
        # IMPORTANT: if multiple actions give the same Q-value, choose the largest action number for the policy. 
        # HINT: We only compute policies for states in stateActions.
        opt_state_action = {} # AKA Optimum Policy
        for state, actions in stateActions.items():
            # comparison function
            def qval(action):
                return (computeQ(V, state, action), action)
            
            # max() takes in a tuple and as seen above the first item in the tuple is Q_val,
            # if the Q_vals of two actions are same, max will automatically compare the second item i.e. the action number
            opt_state_action[state] = max(actions, key=qval)
        
        return opt_state_action


    print('Running valueIteration...')
    
    # STEP 1: Find the values at each state using Value Iteration
    V = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
        # update V values using the computeQ function above.
        # repeat until the V values for all states do not change by more than epsilon.
        for state in stateActions.keys():
            # We're finding optimum value, so taking max here
            newV[state] = max([computeQ(V, state, action) for action in stateActions[state]])

        if all(abs(newV[state] - V[state]) < epsilon for state in stateActions.keys()):
            break

        V = newV
        numIters += 1

    V_opt = newV
    print(("valueIteration: %d iterations" % numIters))

    # STEP 2: Find optimal policy at each state using the true V(s) found for all states
    opt_policy = computePolicy(V_opt)
    return opt_policy

############################################################
# Problem 3b
# Model-Based Monte Carlo

# Runs value iteration algorithm on the number line MDP
# and prints out optimal policy for each state.
def run_VI_over_numberLine(mdp: util.NumberLineMDP):
    succAndRewardProb = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, 0.2, mdp.penalty), (-mdp.n, 0.8, mdp.leftReward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, 0.3, mdp.penalty), (-mdp.n, 0.7, mdp.leftReward)],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.rightReward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.rightReward)]
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [(s+1, 0.2, mdp.penalty), (s - 1, 0.8, mdp.penalty)]
        succAndRewardProb[(s, 2)] = [(s+1, 0.3, mdp.penalty), (s - 1, 0.7, mdp.penalty)]

    pi = valueIteration(succAndRewardProb, mdp.discount)
    return pi


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, calcValIterEvery: int = 10000,
                 explorationProb: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]

    # Epsilon greedy exploration for Model Based Monte-Carlo
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 1e5 + 1)
        
        # coin_toss
        c_toss = random.random()
        # Cases when we choose to explore
        if (c_toss < explorationProb and explore) or (state not in self.pi.keys()):
            return random.choice(self.actions)
        # Cases when we need to use policy
        else:
            return self.pi[state]

    # We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
    # For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        """
        Here we do two things:
        1. Estimate the state transition probabilities (T) and rewards (R) every k MonteCarlo iterations
        2. Using updated (s, a, s', r, t) we find the optimal policy using value iteration
        """
        
        # caching our experiments
        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            """
            Estimate succAndRewardProb based on self.tCounts and self.rTotal.
            Hint 1: prob(s, a, s') = (counts of transition (s,a) -> s') / (total transtions from (s,a))
            Hint 2: Reward(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
            Then run valueIteration and update self.pi.
            """
            # succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
            succAndRewardProb = defaultdict(list)

            # Loop through every (state, action) pair we've every seen
            for (s,a), next_state_count in self.tCounts.items():

                # How many times did we take action 'a'. We know tCounts maps self.tCounts[(state, action)][nextState] += 1
                num_transitions = sum(next_state_count.values())

                # For every new state that resulted from the same action, update the reward and transition probability
                for next_state, count in next_state_count.items():
                    state_transition_prob = count / num_transitions
                    mean_reward = self.rTotal[(s, a)][next_state] / count

                    succAndRewardProb[(s,a)].append((next_state, state_transition_prob, mean_reward))

            # Remember this function is called for a particular state pair. We need to find best policy for this state 
            self.pi = valueIteration(succAndRewardProb, self.discount)

    """
    Drawbacks of model based Monte Carlo:

    1. It caches transition probs and reward for every state-action pair. If our state space is continous, GG
    2. Running value iteration with inf. states is expensive 
       (which basically looks at one state at a time and finds the best Q val(s, a) for that s)
    3. Chunking continuous state into discrete bins is an approximation
    4. Because agent only plans based on what it's seen, if we don't explore enough may not be optimal.
       i.e. No aspect of hypothesizing what could be a good action based on reasoning 
    """
            

############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ) # dict mapping [state, action] -> reward
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT 1: You can access Q-value with self.Q[state, action]
    # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        # explore
        coin_toss = random.random()
        if (coin_toss < explorationProb and explore):
            return random.choice(self.actions)
        # return optimal action
        else:
            def get_qval(action):
                return self.Q[state, action]
            return max(self.actions, key=get_qval)


    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize() 
    # HINT 1: The target V for the current state is a combination of the immediate reward
    # and the discounted future value.
    # HINT 2: V for terminal states is 0
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        """
        How it works: Instead of learning the value of a state V(s), it learns the Q_val Q(s,a). 
        
        The Math (Temporal Difference Update):
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)]

        or

        - Q(s,a) <- Q(s,a) + alpha * [max_Q_val - Q(s,a)]

        - max_Q_val = (current_reward + discounted_future_reward)

        or 

        - new_Q_val = current_Q_val + step_size*(max_Q_val - current_Q_val)
        """
        # Calculate target (current_reward + discount*max_future_reward)
        old_Q_val = self.Q[state, action]
        current_reward = reward

        if terminal:
            max_future_reward = 0.0
        else:
            max_future_reward = max([self.Q[nextState, a] for a in self.actions])
            
        target = current_reward + (self.discount*max_future_reward)
        error = target - old_Q_val

        # Q(s,a) <- Q(s,a) + step_size * error
        self.Q[state, action] = self.Q[state, action] + self.getStepSize() * error

############################################################
# Problem 4b: Fourier feature extractor
def fourierFeatureExtractor(
        state: StateT,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]

    This can be broken down to first get (showing the broadcasted addition of x and y terms):
    
    1. curr_sum (x terms with scale 2): [0, 2x, 4x]
    2. new_term (y terms with scale 1): [0, y, 2y]
    
    3. Reshape curr_sum to a column vector (3x1):
       [[0],
        [2x],
        [4x]]
        
    4. Reshape new_term to a row vector (1x3):
       [[0, y, 2y]]
       
    5. Add them together. NumPy broadcasts (stretches) them to match as a 3x3 matrix:
       [[ 0+0,    0+y,    0+2y  ],
        [ 2x+0,   2x+y,   2x+2y ],
        [ 4x+0,   4x+y,   4x+2y ]]
    
    (This matrix is flattened, and the process repeats to add the z terms).

    Then flattened to get
    [1, 2x, y, z, 4x, 2y, 2z, 4x + y, 4x + z, y + z, 4x + 2y, 4x + 2z, ..., 4x + 2y + 2z]

    Finally, apply cosine and multiply by pi to get
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]

    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((maxCoeff+1)^(len(state)),).
    #
    # HINT: refer to util.polynomialFeatureExtractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # Create the base coefficients: [0, 1, 2, ..., maxCoeff]
    coeffs = np.arange(maxCoeff + 1)
    
    # Initialize the first term: c_0 * s_0
    curr_sum = coeffs * (state[0] * scale[0])

    # Iteratively add the combinations for the remaining state dimensions
    for i in range(1, len(state)):
        # Create the new term: c_i * s_i
        new_term = coeffs * (state[i] * scale[i])

        # Broadcast addition to get all combinations of (curr_sum + new_term)
        # Shape: (len(curr_sum), 1) + (1, maxCoeff + 1) -> (len(curr_sum), maxCoeff + 1)
        curr_sum = curr_sum.reshape((len(curr_sum), 1)) + new_term.reshape((1, maxCoeff + 1))

        # Flatten for the next iteration
        curr_sum = curr_sum.flatten()

    # Finally, apply the cosine function to the entire array of sums
    features = np.cos(np.pi * curr_sum)

    return features

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, explorationProb=0.2):
        '''
        - featureDim: the dimensionality of the output of the feature extractor
        - featureExtractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.featureDim = featureDim
        self.featureExtractor = featureExtractor
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.W = np.random.standard_normal(size=(featureDim, len(actions)))
        self.numIters = 0

    def getQ(self, state: np.ndarray, action: int) -> float:
        """
        Args:
            state : The state at which we want to find Q value
            action : The action at which we want to find Q value
        
        Returns:
            Q(state, action)

        NOTE: Previously there were finite (state, action) pairs and we could store Q values in a dictionary
        Now, due to continuous state space, instead of querying `self.Q[state, action]`, we do `Q = action @ feature`
        """
        # Return Q(state, action) based on self.W and self.featureExtractor(state)
        features = self.featureExtractor(state) # shape: (featureDim,)
        # self.W shape: (featureDim, num_actions)
        W_a = self.W[:, action] # shape: (featureDim,)

        # Q_val = W_a (featureDim, ) @ features (featureDim,)
        return np.dot(W_a, features) # output_shape = 

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT: This function should be the same as your implementation for 4a.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        
        # BEGIN_YOUR_CODE
        coin_toss = random.random()
        if coin_toss < explorationProb:
            return np.random.choice(self.actions)
        else:
            return max(self.actions, key=lambda x: self.getQ(state, x))

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.numIters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update W using self.getStepSize()
    # HINT 1: this part will look similar to 4a, but you are updating self.W
    
    """
    Function approximation update rule:


    """
    def incorporateFeedback(self, state: np.ndarray, action: int, reward: float, nextState: np.ndarray, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        old_Q_val = self.getQ(state, action)
        current_reward = reward

        if terminal:
            max_discounted_future_reward = 0
        else:
            max_discounted_future_reward = self.discount * max([self.getQ(nextState, a) for a in self.actions])
        
        target = current_reward + max_discounted_future_reward
        error = target - old_Q_val

        # NOTE: self.W[:, action].shape = (featureDim,)
        # NOTE: So to update that we get the the error in magnitude and multiply with the feature vector given by self.featureExtractor(state)
        self.W[:, action] = self.W[:, action] + self.getStepSize() * error * self.featureExtractor(state)
        # END_YOUR_CODE

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 explorationProb=0.2):
        super().__init__(featureDim, featureExtractor, actions,
                         discount, explorationProb)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.force,
        mdp1.env.gravity,
        10000,
        explorationProb=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.force,
        mdp2.env.gravity,
        0.065,
        explorationProb=0.2,
    )
    sampleKRLTrajectories(mdp1, rl1)
    sampleKRLTrajectories(mdp2, rl2)

def sampleKRLTrajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_RL_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.startState()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
