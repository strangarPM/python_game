import random  # Import random module for JARVIS's random move selection
from sklearn import tree  # Import tree module from Scikit-learn for Machine Learning predictions
import colorama  # Import colorama for colored terminal output
from colorama import Fore, Style  # Import Fore for colors, Style for reset

# Initialize colorama for cross-platform support
colorama.init()

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

def train_ml_model(player):
    # Train a Machine Learning model to predict the next move based on player's move history
    # player: The Player object (e.g., "Parth") whose moves we’ll use
    # Returns None if not enough data, otherwise returns a trained DecisionTreeClassifier
    if len(player.move_history) < 3:  # Need at least 3 moves for training (2 for input, 1 for output)
        return None  # Not enough data to train, so return None
    
    X = []  # List of input features (past moves) for ML training
    Y = []  # List of target labels (next moves) for ML training
    
    # Create training data: Use last 2 moves to predict the next move
    # Example: If history is [0, 1, 2, 0, 1], use [0, 1] → 2, [1, 2] → 0, [2, 0] → 1
    for i in range(len(player.move_history) - 2):
        X.append([player.move_history[i], player.move_history[i + 1]])  # Last two moves as features
        Y.append(player.move_history[i + 2])  # Next move as the label
    if not X:  # If no data was added (e.g., history too short), return None
        return None
    
    # Create and train a Decision Tree Classifier to learn move patterns
    # DecisionTreeClassifier is simple, interpretable, and good for small datasets
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)  # Train the model on the features (X) and labels (Y)
    return clf  # Return the trained model for predictions

def predict_move(player):
    # Predict the player's next move using the ML model and return JARVIS’s counter move
    # player: The Player object (e.g., "Parth") whose history we’ll predict from
    clf = train_ml_model(player)  # Get or train the ML model
    if clf is None or len(player.move_history) < 2:
        # If no model or insufficient data (< 2 moves for prediction), guess randomly
        print(f"{Fore.BLUE}JARVIS: Not enough data, sir—I’ll guess randomly.{Style.RESET_ALL}")
        return random.choice(["rock", "paper", "scissors"])
    
    # Use the last 2 moves to predict the next move
    last_moves = player.move_history[-2:]  # Get the last 2 moves from history (e.g., [1, 2])
    prediction = clf.predict([last_moves])[0]  # Predict the next move (0, 1, or 2)
    move_map = {0: "rock", 1: "paper", 2: "scissors"}  # Map numerical predictions back to text
    predicted_move = move_map[prediction]  # Convert prediction to text (e.g., "scissors")
    print(f"{Fore.BLUE}JARVIS: Based on your past moves, I predict you’ll pick {predicted_move}, sir!{Style.RESET_ALL}")  # Conversational output, like JARVIS chatting with Tony
    # JARVIS picks a move to counter the predicted move, maximizing his chance of winning
    # e.g., if you’re predicted to pick "rock," JARVIS picks "paper" to win
    counter_move = {"rock": "paper", "paper": "scissors", "scissors": "rock"}[predicted_move]
    return counter_move  # Return JARVIS’s strategic move to beat or tie your predicted move

def check_win(player, computer, player_choice, computer_choice):
    # Determine the winner of a round and update scores accordingly
    # player: The human player (e.g., "Parth")
    # computer: The AI player (e.g., "JARVIS")
    # player_choice, computer_choice: The moves as text ("rock," "paper," "scissors")
    print(f"{Fore.BLUE}\n{player.name} Chose {player_choice},\n {computer.name} Chose {computer_choice}{Style.RESET_ALL}")
    
    if player_choice == computer_choice:
        # If both players pick the same move, it’s a tie
        print(f"{Fore.BLUE}It’s a tie! We’re evenly matched, sir!{Style.RESET_ALL}")
    elif player_choice == "rock":
        if computer_choice == "scissors":
            # Rock beats scissors, so player wins
            print(f"{Fore.BLUE}Rock smashes scissors! You’ve outsmarted me, sir—well done!{Style.RESET_ALL}")
            player.update_score(1)  # Add 1 to player’s score
        else:
            # Paper beats rock, so JARVIS wins
            print(f"{Fore.BLUE}Paper covers the rock! I win this round, sir—impressive strategy!{Style.RESET_ALL}")
            computer.update_score(1)  # Add 1 to JARVIS’s score
    elif player_choice == "paper":
        if computer_choice == "rock":
            # Paper beats rock, so player wins
            print(f"{Fore.BLUE}Paper covers the rock! Victory is yours, sir!{Style.RESET_ALL}")
            player.update_score(1)
        else:
            # Scissors beat paper, so JARVIS wins
            print(f"{Fore.BLUE}Scissors cut the paper! I’ve bested you this time, sir!{Style.RESET_ALL}")
            computer.update_score(1)
    elif player_choice == "scissors":
        if computer_choice == "paper":
            # Scissors beat paper, so player wins
            print(f"{Fore.BLUE}Scissors cut the paper! You’ve triumphed, sir!{Style.RESET_ALL}")
            player.update_score(1)
        else:
            # Rock beats scissors, so JARVIS wins
            print(f"{Fore.BLUE}Rock smashes scissors! I claim this victory, sir!{Style.RESET_ALL}")
            computer.update_score(1)

def show_score(player, computer):
    # Display the current scores for both players
    # player, computer: Player objects with scores to display
    print(f"{Fore.BLUE}\nScoreboard, sir: {player.name} - {player.get_score()}, {computer.name} - {computer.get_score()}{Style.RESET_ALL}")

def play_game():
    # Main game loop to run the Rock, Paper, Scissors game
    # Initializes players, handles rounds, and ends when the player stops
    print(f"{Fore.BLUE}Greetings, sir! I’m JARVIS, ready for a thrilling game of Rock, Paper, Scissors. Shall we begin?{Style.RESET_ALL}")
    player = Player("Parth")  # Create human player
    computer = Player("JARVIS")  # Create AI player (JARVIS)
    
    while True:
        # Run one round of the game
        player_choice = player.choose()  # Get player’s move
        computer_choice = predict_move(player)  # Get JARVIS’s move using ML prediction
        check_win(player, computer, player_choice, computer_choice)  # Determine winner and update scores
        show_score(player, computer)  # Show current scores
        print(f"{Fore.YELLOW}\n{player.name}’s move history: {player.move_history}{Style.RESET_ALL}")
        play_again = input(f"{Fore.BLUE}\nWould you like another round, sir? (yes/no): {Style.RESET_ALL}").lower()  # Ask to continue
        if play_again != "yes":
            # End the game if the player says no
            print(f"{Fore.BLUE}\nFarewell, sir! Final score: {player.name} - {player.get_score()}, {computer.name} - {computer.get_score()}{Style.RESET_ALL}")
            break

# Start the game
play_game()