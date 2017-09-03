# Random number guess game, quit after 10 tries.
import random
number = random.randint(1, 1000)
guess = 0
count=1
while guess != number:
    guess = int(input("please enter a number between 1 and 1000:"))
    if guess == 0:
        print("thanks for playing")
        break
    if int(guess) > int(number):
        if count == 10:
            print('sorry, you have reached maximum ')
            break
        print("please guess lower")
        count += 1
    elif guess < number:
        if count == 10:
            print('sorry, you have reached maximum ')
            break
        print("please guess higher")
        count += 1
    else:
        print(" Congrats, you win")
