import random  # Import random module for JARVIS's random move selection
import numpy as np
from gymnasium.spaces import Discrete
import colorama  # Import colorama for colored terminal output
from colorama import Fore, Style  # Import Fore for colors, Style for reset
import matplotlib.pyplot as plt  # Import matplotlib for visualization

# Initialize colorama for cross-platform support
colorama.init()

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
        print(f"{Fore.BLUE}JARVIS: Not enough data, sir—I’ll guess randomly.{Style.RESET_ALL}")
        return random.choice([0, 1, 2])  # Explore: pick a random move (10% chance)
    # We first get the state row from the Q-table and then we get the index of the maximum value in that row  
    # Get the Q-values for the current state
    q_values = self.q_table[state]
    # Find the maximum Q-value
    max_q = np.max(q_values)
    # Find all actions (indices) with the maximum Q-value
    best_actions = np.argwhere(q_values == max_q).flatten()
    # Randomly pick one action from the best actions (handles ties randomly)
    return random.choice(best_actions)  # Exploit: pick a random move among those with highest Q-value (90% chance)
  
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
    


class Player:
    def __init__(self, name):
        # Initialize a Player object with a name, score, and empty move history
        # name: The player's name (e.g., "Parth" or "JARVIS")
        # score: Tracks wins (starts at 0)
        # move_history: List to store past moves as numbers (0=rock, 1=paper, 2=scissors) for ML training
        self.name = name
        self.score = 0
        self.move_history = []  # Store up to 5 moves to train the ML model
    
    def update_score(self, points):
        # Update the player's score by adding points (e.g., +1 for a win)
        # points: Number of points to add (typically 1 for a win)
        self.score += points
    
    def get_score(self):
        # Return the player's current score
        # Used to display scores during the game
        return self.score
    
    def choose(self):
        # Handle player or JARVIS move selection, store it in move_history, and return the text choice
        options = ["rock", "paper", "scissors"]  # Valid move options for the game
        move_map = {"rock": 0, "paper": 1, "scissors": 2}  # Map text moves to numbers for ML
        if self.name == "JARVIS":
            # If the player is JARVIS, randomly select a move
            # This mimics JARVIS's initial random behavior before ML predictions
            choice = random.choice(options)
        else:
            # If the player is human (e.g., "Parth"), prompt for input until a valid move is entered
            while True:
                choice = input(f"{self.name}, enter your choice (rock, paper, scissors): ").lower()
                if choice in options:
                    # If the input is valid ("rock", "paper", or "scissors"), break the loop
                    break
                print(f"{Fore.BLUE}Invalid choice, sir! Please enter rock, paper, or scissors.{Style.RESET_ALL}")
        # Convert the text choice to a number for ML and store it in move_history
        numeric_move = move_map[choice]
        self.move_history.append(numeric_move)  # Add the numerical move to history
        if len(self.move_history) > 5:  # Limit history to last 5 moves for ML efficiency
            # Remove the oldest move (index 0) to keep only the most recent 5 moves
            self.move_history.pop(0)
        print(f"{Fore.YELLOW}{self.name}’s move added to history: {numeric_move} (History: {self.move_history}){Style.RESET_ALL}")  # Debug: Show history for verification
        return choice  # Return the text choice for the game logic

def predict_move(player):
  # Predict the player's next move using RL (Q-learning) and return JARVIS’s move
  # player: The Player object (e.g., "Parth") whose history we’ll use as the state
  global rl_agent  # Use global to persist the RL agent across rounds
  if 'rl_agent' not in globals():
    rl_agent = RLAgent()  # Create the RL agent if it doesn’t exist yet
  # Use your last move as the state (0=rock, 1=paper, 2=scissors)
  state = player.move_history[-1] if player.move_history else 0  # Default to 0 (rock) if no history
  print(f"{Fore.YELLOW}Getting your current move: {state}{Style.RESET_ALL}")
  # JARVIS chooses a move using RL (Q-learning)
  jarvis_move = rl_agent.choose_action(state)
  move_map = {0: "rock", 1: "paper", 2: "scissors"}
  jarvis_text_move = move_map[jarvis_move]
  return jarvis_text_move

def check_win(player, computer, player_choice, computer_choice):
    # Determine the winner of a round and update scores accordingly
    # player: The human player (e.g., "Parth")
    # computer: The AI player (e.g., "JARVIS")
    # player_choice, computer_choice: The moves as text ("rock," "paper," "scissors")
    print(f"{Fore.BLUE}\n{player.name} Chose {player_choice},\n {computer.name} Chose {computer_choice}{Style.RESET_ALL}")
    reward_increment = 1
    reward_decrement = -1
    reward = 0  # Default to tie
    if player_choice == computer_choice:
        # If both players pick the same move, it’s a tie
        print(f"{Fore.BLUE}It’s a tie! We’re evenly matched, sir!{Style.RESET_ALL}")
    elif player_choice == "rock":
        if computer_choice == "scissors":
            # Rock beats scissors, so player wins
            print(f"{Fore.BLUE}Rock smashes scissors! You’ve outsmarted me, sir—well done!{Style.RESET_ALL}")
            player.update_score(1)  # Add 1 to player’s score
            reward = reward_decrement
        else:
            # Paper beats rock, so JARVIS wins
            print(f"{Fore.BLUE}Paper covers the rock! I win this round, sir—impressive strategy!{Style.RESET_ALL}")
            computer.update_score(1)  # Add 1 to JARVIS’s score
            reward = reward_increment
    elif player_choice == "paper":
        if computer_choice == "rock":
            # Paper beats rock, so player wins
            print(f"{Fore.BLUE}Paper covers the rock! Victory is yours, sir!{Style.RESET_ALL}")
            player.update_score(1)
            reward = reward_decrement
        else:
            # Scissors beat paper, so JARVIS wins
            print(f"{Fore.BLUE}Scissors cut the paper! I’ve bested you this time, sir!{Style.RESET_ALL}")
            computer.update_score(1)
            reward = reward_increment
    elif player_choice == "scissors":
        if computer_choice == "paper":
            # Scissors beat paper, so player wins
            print(f"{Fore.BLUE}Scissors cut the paper! You’ve triumphed, sir!{Style.RESET_ALL}")
            player.update_score(1)
            reward = reward_decrement
        else:
            # Rock beats scissors, so JARVIS wins
            print(f"{Fore.BLUE}Rock smashes scissors! I claim this victory, sir!{Style.RESET_ALL}")
            computer.update_score(1)
            reward = reward_increment
    
    move_map = {"rock": 0, "paper": 1, "scissors": 2}
    player_choice_num = move_map[player_choice]
    computer_choice_num = move_map[computer_choice]
    next_state = player_choice_num
    rl_agent.update_q_table(player_choice_num, computer_choice_num, reward, next_state)  # Update Q-table after the round

def show_score(player, computer):
    # Display the current scores for both players
    # player, computer: Player objects with scores to display
    print(f"{Fore.BLUE}\nScoreboard, sir: {player.name} - {player.get_score()}, {computer.name} - {computer.get_score()}{Style.RESET_ALL}")

def plot_move_history(player, computer):
    # Plot the move history for both players to visualize patterns
    # player, computer: Player objects with move_history
    moves = {0: "Rock", 1: "Paper", 2: "Scissors"}  # Map numbers to readable names for plotting
    player_moves = [moves[move] for move in player.move_history]
    computer_moves = [moves[move] for move in computer.move_history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(player_moves, label="Parth’s Moves", color="blue", marker="o")
    plt.plot(computer_moves, label="JARVIS’s Moves", color="red", marker="x")
    plt.title("Rock, Paper, Scissors Move History")
    plt.xlabel("Move Number")
    plt.ylabel("Move")
    plt.legend()
    plt.grid(True)
    plt.show()

def play_game():
    # Main game loop to run the Rock, Paper, Scissors game
    # Initializes players, handles rounds, and ends when the player stops
    print(f"{Fore.BLUE}Greetings, sir! I’m JARVIS, ready for a thrilling game of Rock, Paper, Scissors. Shall we begin?{Style.RESET_ALL}")
    player = Player("Parth")  # Create human player
    computer = Player("JARVIS")  # Create AI player (JARVIS)
    
    while True:
        # Run one round of the game
        computer_choice = predict_move(player)  # Get JARVIS’s move using ML prediction
        player_choice = player.choose()  # Get player’s move
        check_win(player, computer, player_choice, computer_choice)  # Determine winner and update scores
        show_score(player, computer)  # Show current scores
        print(f"{Fore.YELLOW}\n{player.name}’s move history: {player.move_history}{Style.RESET_ALL}")
        play_again = input(f"{Fore.BLUE}\nWould you like another round, sir? (yes/no): {Style.RESET_ALL}").lower()  # Ask to continue
        if play_again != "yes":
            # End the game if the player says no
            print(f"{Fore.BLUE}\nFarewell, sir! Final score: {player.name} - {player.get_score()}, {computer.name} - {computer.get_score()}{Style.RESET_ALL}")
            plot_move_history(player, computer)  # Plot move history after the game ends
            break

# Start the game
play_game()