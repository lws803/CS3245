import nltk

query = "test AND set"
query2 = "test AND NOT set"
query3 = "test AND (set OR cool)"
query4 = "bill OR Gates AND (vista OR XP) AND NOT mac"

stack = []
queue = []
operators = {"NOT": 3, "AND": 4, "OR": 5}


# Shuting yard algorithm
for token in nltk.word_tokenize(query4):
    # print token
    if (token == "("):
        stack.append(token)
    elif (token == ")"):
        while (stack[-1] != "("):
            queue.append(stack.pop())
        stack.pop()

    elif (token in operators):
        if (len(stack) != 0 and stack[-1] != "(" and operators[stack[-1]] < operators[token]):
            queue.append(stack.pop())
            queue.append(token)
        else:
            stack.append(token)
    else:
        queue.append(token)


while (len(stack) > 0):
    queue.append(stack.pop())


# Create a postfix processor for this
for item in queue:
    print item
