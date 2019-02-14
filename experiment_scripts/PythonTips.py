## Some tips for python programming

# Slicing
'hello'[1:4] # prints 'ell'

# Splitting
'a_b_c'.split('_') # ['a','b','c']

str1.startswith(str2) # Check whether str1 starts with str2

str1.find(str2) # get the position of str2 in str1, returns -1 if cant find

# Printing
var = 10
print 'test is ', var # to chain and print var


# Mutable vs immutable types
# Lists are mutable, but strings and tuples are immutable


# List methods
list0 = list1 + list2 # This creates a new list which containts list1 and list2
list1.append(1) # Appends '1' to list1
list1.extend([1,2]) # Extends list1 with another list
list1.count(0) # Count the number of occurences of 0
list1.index(x) # Find the first location of an element in the list
s.remove(x) # remove x from list
s.sort() # sorts a list
s.reverse() # reverses a list


# Referencing elements in python
list1 = [1,2,3]
list2 = list1 # Does not actually create a new copy of list1. It actually references list1


# OOP in python

class TestClass(DerivedClass1, DerivedClass2, ...):
    # constructor
    def __init__(self):
        pass

    # deconstructor
    def __del__(self):
        pass


class SchoolMember:
    '''Represents any school member.'''
    def __init__(self, name, age):
        self.name = name
        self.age = age
        print '(Initialized SchoolMember: %s)' % self.name

    def tell(self):
        '''Tell my details.'''
        print 'Name:"%s" Age:"%s"' % (self.name, self.age),

# Teacher and student both inherit from SchoolMember
class Teacher(SchoolMember):
    '''Represents a teacher.'''
    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print '(Initialized Teacher: %s)' % self.name

    def tell(self):
        SchoolMember.tell(self) # We pass self here
        print 'Salary: "%d"' % self.salary

class Student(SchoolMember):
    '''Represents a student.'''
    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print '(Initialized Student: %s)' % self.name

    def tell(self):
        SchoolMember.tell(self)
        print 'Marks: "%d"' % self.marks



# Exception handling

while True:
    
    try:
        x = int(raw_input("Please enter a number: "))
        break
    
    except ValueError:
        print "That was not a valid number. Try again..."


# Standard library
import sys
sys.argv # list of arguments while executing the python script

# OS module
import os

os.listdir() # Get all file names in the specified directory



# Aggregating function arguments

def function (a, *args):
    print a
    print args


function(1, 1, 2, 3) # will print 1, [1,2,3]
