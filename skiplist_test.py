import math

class Node:
    def __init__ (self, value, skip_index=None):
        self.value = value
        self.skip_index = skip_index

    def getValue (self):
        return self.value

    def skip(self):
        return self.skip_index


# Condition: Data input must be sorted
class SkipList:
    def __init__ (self, data = []):
        self.skip_list = []

        length = len(data)
        skip_length = math.pow(length, 0.5)
        skip_length = int(skip_length)

        curr_skip = 0
        for i in range(0, length):
            if (i == curr_skip):
                self.skip_list.append(Node(data[i], i+skip_length))
                curr_skip = i + skip_length
            else:
                self.skip_list.append(Node(data[i]))
        
    def get(self, index):
        if (index >= len(self.skip_list)):
            return -1 # Returns -1 to show out of bounds
        return self.skip_list[index].getValue()

    def getSkipIndex(self, index):
        if (index >= len(self.skip_list)):
            return -1 # Returns -1 to show out of bounds

        return self.skip_list[index].skip()

    def getLength (self):
        return len(self.skip_list)


    def find(self, value, start_index=0):
        if (start_index >= len(self.skip_list)):
            return -1
         
        for i in range(start_index, len(self.skip_list)):
            if (self.skip_list[i].getValue() == value):
                return i

            elif (self.skip_list[i].skip() != None and self.skip_list[i].skip() < len(self.skip_list)):
                skip_index = self.skip_list[i].skip()

                # IF skip value is less than the one we want then move iterator to this skip index
                if (self.skip_list[skip_index].getValue() < value):
                    print "skipped", i
                    i = skip_index

                # IF the skip value is the one we want then just return that
                elif (self.skip_list[skip_index].getValue() == value):
                    print "skipped", i
                    return skip_index


if __name__ == "__main__":
    myList = SkipList([1,2,3,4,5,6,7,8,9,10])

    # print myList.get(9)
    # for i in range(0, myList.getLength()):
    #     print myList.getSkipIndex(i)

    print (myList.find(10, 0))

