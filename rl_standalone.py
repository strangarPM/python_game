import random
import numpy as np
from gymnasium.spaces import Discrete
import colorama  # Import colorama for colored terminal output
from colorama import Fore, Style  # Import Fore for colors, Style for reset


# Initialize colorama for cross-platform support
colorama.init()

# Define the Rock, Paper, Scissors environment (simplified, no Gymnasium dependency needed for now)

class RockPaperScissorsEnv:
  def __init__(self):
    # Define action space: 0=rock, 1=paper, 2=scissors
    self.action_space = Discrete(3) # JARVIS’s possible moves
    self.observation_space = Discrete(3) # Your possible moves (states)
    self.player_move = None # Store your move (random for now)
    self.jarvis_move = None # Store JARVIS’s move
    
  def reset(self):
    # Reset the environment for a new game
    self.player_move = None
    self.jarvis_move = None 
    return random.choice([0,1,2]) # Return a random initial state (your move)
  
  def step(self, action):
    # Action is JARVIS’s move (0=rock, 1=paper, 2=scissors)
    # Simulate your move (random, as you’ll play randomly)
    self.player_move = random.choice([0,1,2])# Your random move
    self.jarvis_move = action # JARVIS’s move
    # Calculate reward based on Rock, Paper, Scissors rules
    # +1 for JARVIS win, 0 for tie, -1 for JARVIS loss
    if self.player_move == self.jarvis_move:
      reward = 0 #tie
    elif (self.player_move == 0 and self.jarvis_move == 2) or\
         (self.player_move == 1 and self.jarvis_move == 0) or\
         (self.player_move == 2 and self.jarvis_move == 1):
      reward = -1 # You win, JARVIS loses
    else:
      reward = 1 # JARVIS wins
      
    # Next state is your current move (for simplicity, we use your move as the state)
    next_state = self.player_move 
    done = True # Each game is one round (simplified for now)
    info = {"player_move": self.player_move, "jarvis_move": self.jarvis_move}
    return next_state, reward, done, info
      
# Q-learning Agent for JARVIS
class RLAgent:
  def __init__(self):
      # Initialize Q-table for Q-learning (3 states: your moves, 3 actions: JARVIS’s moves)
      # 3x3 table (0=rock, 1=paper, 2=scissors for states and actions), starting with zeros
      self.q_table = np.zeros((3, 3))
      self.learning_rate = 0.1  # How much new info affects Q-values (low for gradual learning)
      self.discount_factor = 0.9  # How much future rewards matter (high for long-term focus)
      self.epsilon = 0.1  # Chance of exploring (random move) vs. exploiting (best move)


  def choose_action(self, state):
    # Choose an action using epsilon-greedy policy
    # state: Your last move (0=rock, 1=paper, 2=scissors)
    # random.uniform(0, 1) generate a random float between 0 and 1 and if it is less than epsilon(0.1), then it will explore (10% chance to be true)
    if random.uniform(0, 1) < self.epsilon:
        return random.choice([0, 1, 2])  # Explore: pick a random move (10% chance)
    # We first get the state row from the Q-table and then we get the index of the maximum value in that row  
    return np.argmax(self.q_table[state])  # Exploit: pick the move with highest Q-value (90% chance)
  
  def update_q_table(self, state, action, reward, next_state):
    # Update Q-table using Q-learning formula: Q(s,a) = Q(s,a) + α[R + γ*max(Q(s’,a’)) - Q(s,a)]
    # state: Current state (your last move, e.g., 0 for rock)
    # action: JARVIS’s move (0, 1, or 2)
    # reward: Outcome (+1 for win, 0 for tie, -1 for loss)
    # next_state: Your next move (your current move, e.g., 1 for paper)
    current_q = self.q_table[state, action]  # Current Q-value for this state-action pair
    next_max_q = np.max(self.q_table[next_state])  # Best future Q-value for the next state
    new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
    # Update Q-table: Add learning_rate times the difference between current Q and new expected value
    self.q_table[state, action] = new_q
    


  #Why do we pass the object?
  #what is np.argmax()?, What is the np.max()?
  
  
  # Train and test the RL agent
def train_and_test_rl():
    # Create environment and agent
    env = RockPaperScissorsEnv()
    agent = RLAgent()
    
    # Train for 100 episodes (automated, not manual play)
    episodes = 100
    for episode in range(episodes):
        state = env.reset()  # Start a new game with a random state (your move)
        done = False
        while not done:
            # JARVIS assume your move first base on that he chooses his move
            action = agent.choose_action(state)  # JARVIS chooses a move (explore/exploit)
            next_state, reward, done, info = env.step(action)  # Play the game, get reward
            agent.update_q_table(state, action, reward, next_state)  # Learn from the outcome
            state = next_state  # Update state for the next move
    
    # Test against one random move from you
    print("\nTraining complete! Testing JARVIS against one random move...")
    state = env.reset()  # Reset for test
    your_move = random.choice([0, 1, 2])  # Your random move (simulating your play)
    jarvis_move = agent.choose_action(your_move)  # JARVIS’s learned move (no updates during test)
    move_map = {0: "rock", 1: "paper", 2: "scissors"}
    print(f"You (randomly) picked: {move_map[your_move]}")
    print(f"JARVIS (RL) picked: {move_map[jarvis_move]}")

    # Calculate and print the outcome
    if your_move == jarvis_move:
        print(f"{Fore.BLUE}It’s a tie! We’re evenly matched, sir!{Style.RESET_ALL}")
    elif (your_move == 0 and jarvis_move == 2) or \
         (your_move == 1 and jarvis_move == 0) or \
         (your_move == 2 and jarvis_move == 1):
        print(f"{Fore.BLUE}You’ve outsmarted me, sir—well done!{Style.RESET_ALL}")
    else:
        print(f"{Fore.BLUE}I claim this victory, sir!{Style.RESET_ALL}")


if __name__ == "__main__":
    train_and_test_rl()
        
        