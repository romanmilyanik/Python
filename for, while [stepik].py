# https://stepik.org/course/67/syllabus
x = "variable"

# 1. while
a = 5
while a > 0:
    print(a, end = " ")
    a -= 1
    
# result: 5 4 3 2 1 
    
# break      - break the cycle
# continue   - transition to the next iteration



# 2. for
    
# a.
    for i in 2, 3, 5:
        print(i * i, end = " ")
        
    # result: 4 9 25 

# b.
    for i in range(10):
        print(i * i, end = " ")

    # result: 0 1 4 9 16 25 36 49 64 81

# range(10)         - sequence 0 to 9
# range(5)          - [0, 1, 2, 3, 4]
# range(2, 15, 4)   - [2, 6, 10, 14]   2 to 14 with step 4

# range() use in cycle
# to get the sequence use list()
    list(range(5))  # [0, 1, 2, 3, 4]

# "\t"    - tabs
