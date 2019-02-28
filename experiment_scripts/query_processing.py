import nltk

query = "test AND set"
query2 = "test AND NOT set"
query3 = "Matthew AND NOT (Wilson AND Aaryam)"
query4 = "(A AND B AND NOT NOT NOT C) OR D OR E OR NOT NOT G AND (F OR H)"

stack = []
queue = []
operators = {"NOT": 5, "AND": 4, "OR": 3}

# TODO: change this for loop to iterate by index instead
# TODO: Find a way to deal with AND NOT/ OR NOT

# Shunting yard algorithm
for token in nltk.word_tokenize(query4):
    # print token
    if (token == "("):
        stack.append(token)
    elif (token == ")"):
        while (stack[-1] != "("):
            queue.append(stack.pop())
        stack.pop()

    elif (token in operators):
        if (len(stack) != 0 and stack[-1] != "(" and operators[stack[-1]] <= operators[token]):
            if (stack[-1] == "NOT" and token == "NOT"):
                stack.pop()
            elif (queue[-1] == "NOT" and token == "NOT"):
                queue.pop()
            else:
                queue.append(token)
        else:
            stack.append(token)
    
    # Term
    else:
        queue.append(token)


while (len(stack) > 0):
    queue.append(stack.pop())


# Create a postfix processor for this
for item in queue:
    print (item)
