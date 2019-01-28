import random

human_score = 0
computer_score = 0

choices = ['s', 't', 'p']
dict_choices = {'s':0, 't':1, 'p':2}


while (True):
    human_choice = raw_input("Select a choice (s, t, p): ")

    choice = random.choice(choices)

    if (human_choice == 'p' and choice == 's'):
        computer_score += 1
    elif (choice == 'p' and human_choice == 's'):
        human_score += 1

    elif (dict_choices[choice] < dict_choices[human_choice]):
        human_score += 1
    elif (dict_choices[choice] > dict_choices[human_choice]):
        computer_score += 1
    else:
        pass


    print 'human computer'
    print human_choice, ' ', choice
    print human_score, ' ', computer_score

