# Introduction
Reinforcement Learning (RL) sets a different stage compared to other types of machine learning like supervised and unsupervised learning. 
In RL, there is an agent and there is an environment.
Agent interacts with the environment by taking actions and in turn receiving reward from the environment.

Agent observes the ```state``` of the environment and based on that ```state``` it takes an ```action```.
As a result the ```state``` of the environment changes, and the agent simultaneously receives ```reward``` from the ```environment``` and observes the new ```state``` of the environment.
The goal of Reinforcement Learning is for the ```agent``` to learn a patterns of ```action```s that maximize total ```reward```

We can take a robot learning to walk as an example of reinforcement learning.
The robot is set in an environment, it can be a physical land or a simulator. Robot can take actions like moving its legs, feet, joints. As it takes these actions, the state of the environment changes depending on how far it moved.
Reward is set to a distance the robot moved. So in order to walk as long distance as possible it should learn to walk efficiently. We won't teach the robot how to walk. It learns the most efficient way of doing so from experience.
Whenever it loses its balance and falls, it will note down the actions that caused this and perhaps try to avoid them.
On the other hand the actions that kept it in balance and allowed it to walk longer distances, it will try to imitate more frequently perhaps.
In a way the robot is running experiments, gathering data and learning best actions based on that data.

> **Note**
It is important not to confuse *reward*s in Reinforcement Learning with *labels* in supervised learning.
A *reward* is a feedback that the agent receives from the environment in response to its action.
A *reward* doesn't necessarily tell the agent that the action it took was the most optimal, the most correct one.
A *label* does tell the machine the right answer. If we were to think of *reward* as a *label* it would be like telling the agent the right course of actions every time it took any action.


# Multi Armed Bandit problem
Let's solve a very basic Reinforcement Learning problem known as **Multi Armed Bandit**.
Although this problem is very simple, it demonstrates essential properties of reinforcement learning.
You can think of **Multi Armed Bandit** as a series of **slot machines**.
Every slot machine has a lever. When you pull the lever, the icons on the screen start rolling and depending on the final pattern of icons on the screen you either receive coins (*rewards*) or not.
In this problem, every slot machine has different odds of spitting out coins.
The agent doesn't know about these odds beforehand. It will learn about them by trying different slot machines multiple times within limited amount of times it is allowed to play, let's say 1000 times.
The goal of the agent is to collect as many coins as possible within those 1000 rounds of playing with slot machines.

Before jumping to the solution of **Multi Armed Bandit** problem, let's list out its components in the context of Reinforcement Learning.
- **Environment** : A series of slot machines is the environment in this case
- **Agent** : The player is the agent.
- **Action** : Choosing a particular slot machine corresponds to action
- **Reward** : The coins received when pulling the lever of a slot machine corresponds to reward

> **Note**
In usual Reinforcement Learning problems, the environment has states and its state changes as the agent takes actions.
In Multi Armed Bandit problem, the player is faced with the same set of slot machines in every round, so the environment state doesn't change at all. But in usual RL problem, the Agent takes action based on reward and the state of the environment. We will learn more about it in ```Markov Decision Process```.

## Which one is a better slot machine
The key to solving a **Multi Armed Bandit** problem is of course to choose the slot machine that spits out more coins.
But a slot machine has some randomness attached to it. The number of coins it spits out is different (random) in every round. It would be no fun if the slot machine were deterministic, i.e. I can tell exactly how many coins it will spit out in every round. Besides the casino running such a slot machine would go bankrupt instantenously.
But I still want to express a difference between slot machines. 
Even if the number of coins spat out in every round is random, it is completely plausible that over a long run slot machine 1 spits out more coins than slot machine 2.
We can quantify randomness with probabilities. We can say for instance that a slot machine 1 spits out coins with 0.6 probability while a slot machine 2 spits out coins with 0.2 probability. Probabilities are always floating point numbers between 0 and 1. In our example, as we play two slot machines the first one will definitely pay us a lot more coins, making us richer and happier. Who knew it was so easy to get happy.

Above example is quite primitive and vague. Let's make our slot machine a little more interesting, a little more complex and define its probability distribution over all patterns of coins it spits out.

1. Our slot machine can spit out 1, 5, 10 or 0 (no) coins in a single round. So when you pull its lever, you either get 1 coin, or 5 coins, or 10 coins (yuu huuu) or no coins (mehhhh). A coin for mommy, a coin for daddy, but no coin for little poor me.
2. There is probability attached to every pattern. The set of probabilities over all possible cases is called **Probability Distribution**. (Don't you feel academic and smart). The sum of probabilities over all possible cases is always equal to one. You are either alive or dead on any given day. So the sum of probabilities of you being alive or dead is equal to 1. Back to our slot machine as we are still alive and interested in hitting jackpot. Below is an example of probability distribution for 2 slot machines

- Probabilities of gaining coins for slot machine 1

| Number of Coins |    0   |   1   |   5   |  10  |
|-----------------|--------|-------|-------|------|
| Probability     |  0.70  | 0.15  | 0.12  | 0.03 |

- Probabilities of gaining coins for slot machine 2

| Number of Coins |   0   |   1   |   5   |  10  |
|-----------------|-------|-------|-------|------|
| Probability     | 0.50  | 0.40  | 0.09  | 0.01 |

Usually probability distribution of slot machines is not known to the player. But let's assume we cheated and we knew these values. Then which slot machine would we choose? (Who knew you still have to do some work after cheating)
Well, even if you knew nothing about the theory of probability, you could already build some strategy by looking numbers. For instance the first slot machine has higher probability of giving us both 5 and 10 coins. So if we are after jackpot should we choose slot machine 1? But how high is good enough though? Probability of 0.03 doesn't look impressive. Besides slot machine 1 has higher probability of returning no coins at all. Probability of 0.7 for zero coins means on average 7 out of 10 rounds it spits out nothing. So perhaps I am better off choosing Slot machine 2?
But what if I told you that there is a formula to tell you precisely which machine pays off higher than the other one over long term? 

There is such a formula and it is absurdly simple. Just multiply number of coins with probabilities associated with them and the slot machine with higher result pays off higher in long term. This calculation is called ```Expected Return```, which literally expresses your average gain in the long run. To be more precise it will tell you how many coins on average will you earn in every round.

So let's quickly calculate ```Expected Return```s of each machine and compare them and find out which one makes us rich.

- Expected Return of Slot machine 1
 0.7 * 0 + 0.15 * 1 + 0.12 * 5 + 0.03 * 10 = 1.05

- Expected Return of Slot machine 2
 0.5*0 + 0.4 * 1 + 0.09 * 5 + 0.01 * 10 = 0.95

 So now finally we know with certainty which slot machine makes us rich in the long run. Above tells us that in every round on average in the long run Slot Machine pays us 1.05 coins while Slot Machine 2 pays us 0.95 coins. Pay attention to the words ```aveage``` and ```in the long run```. These are important. Because these numbers don't guarantee that Slot machine 1 pays more coins if you are playing very few rounds. So my friend, if you are in Vegas and know ```Expected Returns``` of whatever you are gambling, play it ten thousand times, not few hundred times. (Author doesn't take any responsibility for any loss you incur from Gambling. Gambling is dangerous for your wallet.)

 >**Note**
 In the lingo of Reinforcement Learning ```Expected Return``` like above is called ```Value``` or in a sense that the value we derive from an action as ```Action Value```.

## Represent words with symbols
Let's get mathematical and symbolic a little bit. So that people in the academia show us some respect. It is common to represent ```rewards``` with capital ```R```. In our case ```R``` consist of a set ```{0, 1, 5, 10}```. Also to explicitly represent reward received in round ```t``` we use R~t~ notation.
Actions are expressed with ```A```. If we named our slot machines as a and b then our action set would be ```{a,b}```
Expected return or value is expressed as ```E```. ```E[R]``` would read as ```Expected Reward```, ```E[R|A=a]``` would read as ```Expected Reward given action a```

To express the true value of an action we use small case ```q```, to express estimated value of an action (since agent can't know true value of action beforehand) we use uppercase ```Q```

```q(a) = E[R | A = a]```

## Finally algorithm

**No dear, just wait a little more. First learn how to implement Value in Code**

To summarize the situation of the ```Agent``` in the game of Multi Armed Bandit.
- if the player knew expected returns, i.e. the ```Value``` of each slot machine, he would have chosen the slot machine with the highest Value and call it a day

- But the player never knows true ```Value```s of slot machines

- So the player has to estimate or guess as accurately as possible the ```Value```s of slot machines

So one might ask how do I estimate ```Values``` associated with my ```Actions```, in our case the values of slot machines.
The answer is obvious. You play each machine multiple rounds and take the average of coins you gained from each and call that the estimated ```Value``` of playing machine ```a``` or machine ```b```

For instance, let's say you played each machine 3 rounds and the results were as following

| Slot machine | Round 1 | Round 2 | Round 3 |
|--------------|---------|---------|---------|
| a            | 0       | 1       | 5       |
| b            | 1       | 0       | 0       |

In this case


Value of slot machine \( a \):

$$ Q(a) = \frac{(0 + 1 + 5)}{3} = 2 $$

Value of slot machine \( b \):

$$ Q(b) = \frac{(1 + 0 + 0)}{3} = 0.33 $$

The general formula would be

$$ Q = \frac{(R_{1} + R_{2} + R_{3} + ... + R_{n})}{n} $$

Let's write simple code to calculate ```Action Value```


```python
import numpy as np

rewards = []
N=11

for n in range(1,N+1):
    rewards.append(np.random.rand())
    Q = sum(rewards) / n
```

Above code has one drawback. It keeps all rewards across all rounds in the memory by appending them into a list. It would be much more elegant if we could calculate ```Q``` without wasting too much memory, even though the price of memory went down recently.

As it turns out, we can do better with the power of math. All we need is to replace one thing with another.

$$ Q_{n-1} = \frac{(R_{1}+R_{2}+...+R_{n-1})}{n-1} $$
$$ Q_{n-1}*(n-1) = R_{1}+R_{2}+...+R_{n-1}$$

$$ Q_{n} = \frac{(R_{1}+R_{2}+R_{3}+...R_{n-1}+R_{n})}{n}$$

We replace $$ R_{1}+R_{2}+...+R_{n-1}$$ with 
$$ Q_{n-1}*(n-1) $$

and we get

$$ Q_{n} = \frac{1}{n} ((n-1)*Q_{n-1} + R_{n})$$

with some reordering we can express our value in therms of Q~n-1~ and R~n~

$$ Q_{n} = Q_{n-1} + \frac{1}{n}(R_{n} - Q_{n-1})$$

This equation is much more elegant and helps us interpret new ```Value``` as updating ```Previous Value```. As the ```Agent``` goes through playing rounds it updates the estimated action value based on current estimate and the reward it received. We can think of ```1/n``` as learning rate. One way I think about it, just because I get a high reward R~n~ in a given round doesn't mean I should go ahead and update my estimate of Q~n~ to equal R~n~. Instead I act carefully and only update my estimate to certain portion of difference between R~n~ and Q~n-1~. As ```n``` grows to infinity, my action value won't get updated at all because ```1/n``` tends to zero as ```n``` tends to infinity. (There I used the statement from my math textbooks back in my days)

Let's reimplement our value

```python
import numpy as np
N=100
Q = 0
for n in range(1,N+1):
    R = np.random.rand()
    Q = Q + (R - Q) / n
    print(f"Action Value is {Q}")
```
## Agent's strategy
So now you know how to estimate ```Values``` associated with each ```Action```, in our case with choosing a slot machine. We can say the agent can choose one of two strategies

- Always play the slot machine with highest estimated value and never try anything else. This is called ```greedy``` method. Because agent is greedy it sticks to the slot machine that it thinks has highest value. But the greedy agent is also a dumb agent. Because the poor thing doesn't understand that estimated value and true value aren't the same thing. And because it doesn't have courage to try other things, it may end up picking up a loser machine rather than a winner machine, like in the above example. Playing only one round with each machine, if agent falls into trap of thinking ```Slot machine b``` has higher value and keep on playing ```machine b``` only it would have chosen a loser. But let's not criticize our greedy agent too much. In some ways it is being smart and being cautious to take risks. It is prudently exploiting its experience, although it is only one round. Pay attention, another name for this kind of choosing an action is known as ```Exploitation```, because you are ```exploiting``` the knowledge you gained from experience

- Try everything. In other words ```Explore``` what life prepared for you endlessly. This strategy is risk taking and curious. And we know what curiosity did to a cat. To bring us to our case, this strategy is about trying other slot machines without sticking to the slot machine with highest estimated value.

At this point you must have already guessed that the best strategy would be to strike the right balance between ```Exploitation``` and ```Exploration```. And there is such strategy known as ```epsilon - greedy```. Under this strategy the agent will primarily ```Exploit``` its knowledge of the values of slot machines, but once in a while with probability ```epsilon``` it will try other machines. This will give the agent the opportunity to find even better machines if there is one out there.

## Let's finally code
### Environment code
First we will start with implementing the environment as ```Bandit``` class. As we mentioned before collection of slot machines represent the environment. Here we are going to transform our slot machines to very primitive ones where they either give one coin or no coin. We will store probabilities of slotting machines dispensing a coin with certain probabilities into a variable in  ```Bandit``` class. Next ```Bandit``` class will have a ```play``` method where a certain slot machine will be played. This is where ```Bandit``` as an environment will return a reward in response to an action, i.e. picking up a slot machine and playing it.

> **Side note**
We are simulating a random event happening with a certain probability using ```np.random.rand```. In our case random event is a slot machine giving us one coin with certain probability. Let's say we have a slot machine that spits out one coin with 0.7 probability. How can we implement this in code? Well it turns out we can make use of ```np.random.rand``` function. This numpy function draws any real number between 0 and 1 with equal probability. Now if you weren't missing your math classes, you will know that there are inifinite real numbers between 0 and 1. Because it is very hard for our mind to cope with infinity, let's assume there are only 100 distinct numbers between 0 and 1, starting with 0.001, 0.002, 0.003 ... 0.69, 0.70 ... and so on up until 0.98, 0.99 and finally 1.00 and ```np.random.rand``` draws any of these numbers with equal probability. We can calculate this probability quite easily. Because there are 100 distinct numbers, there is 1 in 100 chance to pick up any of these numbers, so the probability of drawing any of these numbers is 1/100=0.01. Now let's ask a question, what's the probability of drawing a number less than 0.7. Well it is equal to adding up probabilities of all the numbers up to 0.7. Because there are 70 distinct numbers up to 0.7 and each of them have 1/100 probability of getting picked up, the sum of their probabilities is equal to 70 * 1/100 = 0.7. We just found out how we can implement an event happening with probability 0.7. All we need is draw a number using ```np.random.rand```, and if that number was less than 0.7, we say that the event happened, if it was greater than 0.7 we say the event didn't happen.

```python
class Bandit:
    def __init__(self,arms=10):
        self.probs = np.random.rand(arms)
    
    def play(self, arm):
        prob = self.probs[arm]
        random_numb = np.random.rand()
        if random_numb < prob:
            return 1
        else:
            return 0
```

### Agent code
Our agent will have to keep track of *how many times it played a slot machine* and *what is the estimated value of a given slot machine*. In more technical terms

- Keep track of how many times an action was picked up from available action space
- Keep track of values associated with each action

We can achieve above by having two lists with length matching action space, i.e. the number of actions agent can choose from. In our case this will equal to number of slot machines our Agent plays. If we have 10 slot machines then both our lists will have length of 10.

Next, the Agent will go through rounds of playing by picking up a slot machine. So our agent has to implement a method to ```choose an action from available action space``` and it also has to implement a method to ```update value associated with that action``` after executing that action and receiving a reward.

```python
class Agent:
    def __init__(self,numb_actions=10, espsilon):
        self.numb_actions = numb_actions
        self.epsilon = espsilon
        self.action_chosen_times = np.zeros(numb_actions)
        self.action_values = np.zeros(numb_actions)
    
    def update_action_value(self, chosen_action, reward):
        self.action_chosen_times[chosen_action] += 1
        self.action_values[chosen_action] += (reward - self.action_values[chosen_action]) / self.action_chosen_times[chosen_action]
    
    def choose_action(self):
        if np.random.rand() < self.epsilon:
            # explore other actions once in a while
            return np.random.randint(0, self.numb_actions)
        # exploit the action with highest estimated value
        return np.argmax(self.action_values)
    
```

## Finally Run The Code. Let the Multi Armed Bandit Rule

Let's place our Agent in the Environment and let it play 1000 rounds. Our goal is to confirm that **Epsilon Greedy Strategy** works. That indeed it allows the Agent to find the best slot machine over time and exploit it as much as possible.

```python
from tqdm import tqdm

numb_actions = 10
epsilon = 0.1
numb_rounds = 1000
environment = Bandit(numb_actions)
agent = Agent(numb_actions, epsilon)

total_reward = 0
total_rewards = []
average_rewards = []

for step in tqdm(range(numb_rounds)):
    chosen_action = agent.choose_action()
    reward = environment.play(chosen_action)
    agent.update_action_value(chosen_action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    average_rewards.append(total_reward/(step+1))
```

Let's confirm if our agent was able to choose the best machine more than others. ```best_machines``` will sort slot machine indices in the descending order of their probabilities of tossing a coin. ```best_machine_probs``` will sort slot machine probabilities of tossing a coin in descending order

```python
best_machines = np.argsort(environment.probs)[::-1]
best_machine_probs = np.sort(environment.probs)[::-1]
```

Now we can print out each best machine index, probability associated with it and how many times it was chosen by the agent

```python
for best_machine, prob in zip(best_machines,best_machine_probs):
    print(f"machine : {best_machine}, its prob: {prob}, how many times it was picked : {agent.action_chosen_times[best_machine]}")
```

We can plot the same as a bar plot.

```python
import matplotlib.pyplot as plt
best_machine_chosen_times=[agent.action_chosen_times[m] for m in best_machines]
best_machine_labels = [str(m) for m in best_machines]

fig,ax=plt.subplots()

ax.bar(best_machine_labels,best_machine_chosen_times)

for i,prob in enumerate(best_machine_probs):
    ax.text(i,best_machine_chosen_times[i]/2,f"{prob:.2f}",ha="center",va="center")

ax.set_xlabel("Slot machines")
ax.set_ylabel("Number of times machine was chosen")
ax.set_title("Slot machine chosen times")
plt.show()
```