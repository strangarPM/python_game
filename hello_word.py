import random


def get_choice():
  player_chose = input("Enter Your Chose (rock, paper, scissors) : ")
  options = ["rock", "paper", "scissors"] 
  computer_chose = random.choice(options)
  choice_dict = {"player" : player_chose, "computer" : computer_chose}
  
  return choice_dict


def check_win(player, computer):
  print(f"You Chose {player},\n Computer Chose {computer}")
  
  if player == computer:
    print("It's a tie!")
  elif player == "rock":
    if computer == "scissors":
      print("Rock smashes scissors! You Won!")
    else:
      print("Paper cover the rock! You lose.")  
  elif player == "paper":
    if computer == "rock":
      print("Paper cover the rock! You won!")
    else:
      print("Scissors cut the paper! You lose.")
  
  
chose = get_choice()
result = check_win(chose["player"], chose["computer"])

print(result)