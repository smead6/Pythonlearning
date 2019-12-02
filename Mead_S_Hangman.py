import random
import string


def choose_word():
    with open("word_list.txt") as txt_file: #open the txt file
        words = txt_file.read().split() # Splits the lines in the file into an array
        selection = random.choice(words) #generates the selection
    #return words #used to test the file has imported the txt file
    return selection # Used to test the selection can return any word from the .txt file and not the first (if using readline)
        


def hangman():
    '''A fun game to help pass some time. It takes a random word from word_list.txt as the word to guess.
    In this case, if you make 7 wrong attempts, you lose the game. No conditions have been set for only taking
    characters of the alphabet as inputs - i.e. '*' will still count as an invalid attempt.
    No 'attempts remaining' counter has been printed for the user. If multiple characters are input, it will not
    penalise the user. Lower and upper cases are also accepted :).'''
    x = 0 # Use to set the attempt counter
    lives_limit = 7 # Inputs a player attempt limit
    word = choose_word()
    #print(word) # Helped make sure I know what I was actually trying to get right
    display_word = "*"*len(word) # generates asterisks for the length of the word
    while word != display_word and x<lives_limit:
        print(display_word)
        go = input( 'Please enter your next guess:')
        attempt=go.lower()
        pos=[]
        for n in range(len(word)): # generates the for loop to index each incident of the attempt in the word
            if word[n]==attempt:
                pos.append(n)
            for i in pos:
                display_word = display_word[:i]+attempt+display_word[i+1:] # similar method to pig latin example
        if attempt not in word: # set condition to take one more life off the player
            x = x+1
        #print(x) Sometimes you need to check how many goes you have left at your own game
        if x== lives_limit: # set lose condition
            print('you lose')   

        if word == display_word: #set win condition
            print('congratulations you win')


hangman()
    
'''def a():
   # A test function to ensure a display function will input a letter if it exists within an input including
#double letters. . It uses the word apple or variation of as it has a double letter. It used a
#character indexing function to find all the locations of that letter if it's in the word.


    pos = [] # set an empty array for all incidents if found in the word
    word = 'appleped' # predetermined to check it works with multiple incidents of a word. 
    print(word) # Helps sanity check the working - too many times I couldn't remember what I'd changed it to.....
    display_word = "*"*len(word) # Generates the word to be shown to the user
    print(display_word) # checks the word being presented to the user

    attempt = input('Have a go ') # user input function
    for n in range(len(word)): # generates the for loop to index each incident of the attempt in the word
        if word[n]==attempt:
            pos.append(n)
        for i in pos:
            display_word = display_word[:i]+attempt+display_word[i+1:] # similar method to pig latin example
    print(display_word)
'''


    

   

    



































    
