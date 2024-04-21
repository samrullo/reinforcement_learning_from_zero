import numpy as np

class Bandit:
    """
    Bandit represents the environment for the agent
    It will initialize the odds of slot machines dispensing a coin    
    """
    def __init__(self,arms=10):
        self.probs = np.random.rand(arms)
    
    def play(self, arm):
        """
        Every time the agent plays a slot machine, it will dispense a coin with its predefined probability
        """
        prob = self.probs[arm]
        random_numb = np.random.rand()
        if random_numb < prob:
            return 1
        else:
            return 0

class NonStationaryBandit:
    """
    Bandit represents the environment for the agent
    It will initialize the odds of slot machines dispensing a coin    
    """
    def __init__(self,arms=10,noise_weight = 0.1):
        self.noise_weight = noise_weight
        self.arms = arms
        self.probs = np.random.rand(arms)
    
    def play(self, arm):
        """
        Every time the agent plays a slot machine, it will dispense a coin with its predefined probability
        Also we add small noise to probabilities of arms
        """
        prob = self.probs[arm]
        self.probs += self.noise_weight * np.random.randn(self.arms)        
        random_numb = np.random.rand()
        if random_numb < prob:
            return 1
        else:
            return 0

class Agent:
    """
    The agent who plays slot machines
    It will choose from available number of actions, which is equivalent to number of slot machines
    """
    def __init__(self,numb_actions=10, espsilon=0.1):
        self.numb_actions = numb_actions
        self.epsilon = espsilon
        self.action_chosen_times = np.zeros(numb_actions)
        self.action_values = np.zeros(numb_actions)
    
    def update_action_value(self, chosen_action, reward):
        """
        After every play, update the value associated with a certain slot machine or certain action
        """
        self.action_chosen_times[chosen_action] += 1
        self.action_values[chosen_action] += (reward - self.action_values[chosen_action]) / self.action_chosen_times[chosen_action]
    
    def choose_action(self):
        """
        In general choose the action with highest value (exploit)
        once in a while with epsilon probability choose action at random (explore)
        """
        if np.random.rand() < self.epsilon:
            # explore other actions once in a while
            return np.random.randint(0, self.numb_actions)
        # exploit the action with highest estimated value
        return np.argmax(self.action_values)

class AlphaAgent:
    """
    The agent who plays slot machines
    It will choose from available number of actions, which is equivalent to number of slot machines
    Also it will weight the newest reward with alpha instead of 1/n, meaning it will give higher weight for newest rewards and exponentially decaying weights to oldest rewards
    """
    def __init__(self,numb_actions=10, espsilon=0.1, alpha=0.1):
        self.numb_actions = numb_actions
        self.epsilon = espsilon
        self.alpha = alpha
        self.action_chosen_times = np.zeros(numb_actions)
        self.action_values = np.zeros(numb_actions)
    
    def update_action_value(self, chosen_action, reward):
        """
        After every play, update the value associated with a certain slot machine or certain action
        """
        self.action_chosen_times[chosen_action] += 1
        self.action_values[chosen_action] += self.alpha * (reward - self.action_values[chosen_action])
    
    def choose_action(self):
        """
        In general choose the action with highest value (exploit)
        once in a while with epsilon probability choose action at random (explore)
        """
        if np.random.rand() < self.epsilon:
            # explore other actions once in a while
            return np.random.randint(0, self.numb_actions)
        # exploit the action with highest estimated value
        return np.argmax(self.action_values)
