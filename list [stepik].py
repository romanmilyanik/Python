# https://stepik.org/course/67/syllabus

city = ["Lviv", "Istanbul", "Kyiv"]

city[0]      # 'Lviv'
city[1]      # 'Istanbul'
city[2]      # 'Kyiv'

len(city)    # 3

city[-1]      # 'Kyiv'
city[-2]      # 'Istanbul'
city[-3]      # 'Lviv'

city[:2]      # ['Lviv', 'Istanbul']
city[::-1]    # ['Kyiv', 'Istanbul', 'Lviv']

# ---------------------------------------------
city1 = ["Lviv", "Istanbul", "Kyiv"]
city2 = ["Lviv", "London"]

city1 + city2       # ['Lviv', 'Istanbul', 'Kyiv', 'Lviv', 'London']
[0, 1, 2] * 3       # [0, 1, 2, 0, 1, 2, 0, 1, 2]
city1[1] = "Rome"   # ['Lviv', 'Rome', 'Kyiv']

city3 = ["Lviv", "Istanbul"]
city3.append("Paris")   # ['Lviv', 'Istanbul', 'Paris']
city3 += ["Oslo"]       # ['Lviv', 'Istanbul', 'Paris', 'Oslo']
city3 += ["NewYork", "Mexico"]  # ['Lviv', 'Istanbul', 'Paris', 'Oslo', 'NewYork', 'Mexico']

city4 =[]   # empty list

city5 = ["Lviv", "Istanbul", "Kyiv"]
city5.insert(1, "Tokyo")   # ['Lviv', 'Tokyo', 'Istanbul', 'Kyiv']

# ---------------------------------------------
# remove
city6 = ["Lviv", "Istanbul", "Lviv"]
city6.remove("Lviv")   # ['Istanbul', 'Lviv']   # remove only first

city7 = ["Lviv", "Istanbul", "Lviv"]
del city7[1]     # ['Lviv', 'Lviv']

# ---------------------------------------------
# search for the element
city8 = ["Lviv", "Istanbul", "Kyiv"]
if "Istanbul" in city8:
    print("in")
    
city9 = ["Lviv", "Istanbul", "Kyiv"]
if "Minsk" not in city8:
    print("out")

ind = city9.index("Lviv")   # 0

# ---------------------------------------------
# sort
city99 = ["Lviv", "Kyiv", "Istanbul"]
ordered_city = sorted(city99)     # ['Istanbul', 'Kyiv', 'Lviv']
city99.sort()       # ['Istanbul', 'Kyiv', 'Lviv']
city99.reverse()    # ['Lviv', 'Kyiv', 'Istanbul']

r = reversed(city99)   # doesnt work ???
t = city99[::-1]       # ['Istanbul', 'Kyiv', 'Lviv']

# ---------------------------------------------
# list definition
a = [1, "A", 2]
b = a
a[0] = 42
# a: [42, "A", 2]
# b: [42, "A", 2]
b[2] = 30
# a: [42, "A", 30]
# b: [42, "A", 30]

# ---------------------------------------------
# list generation
[0] * 5                  # [0, 0, 0, 0, 0]
[0 for i in range(5)]    # [0, 0, 0, 0, 0]
[i*i for i in range(5)]  # [0, 1, 4, 9, 16]

g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
g[1]                     # [4, 5, 6]
g[1][1]                  # 5

n = 4
h = [[0] * n for i in range(n)]                 # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
k = [[0 for j in range(n)] for i in range(n)]   # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
