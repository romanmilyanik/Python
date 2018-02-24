# https://stepik.org/course/67/syllabus
name = "roma"

name[0]    # r
name[1]    # o
name[2]    # m
name[3]    # a
name[-1]   # a
name[-2]   # m
name[-3]   # o
name[-4]   # r

name[1] = "m"   # error!!!

# --------------------------------------
name = "roman"
for c in name:
    print(c)

# result: [r o m a n]

# --------------------------------------
name = "Ololo"
print(name.count("o"))   # 2

# --------------------------------------
s = "aTGcc"
p = "cc"

s.upper()             # 'ATGCC'
s.lower()             # 'atgcc'
s.count(p)            #  1
s.find(p)             #  3
s.find("A")           # -1 (no A)
s.replace("c", "C")   # 'aTGCC'

s = "agTtcAGtc"
s.upper().count("gt".upper())   # 2

# Slicing ==============================
dna = "ATTCGGAGCT"
dna[1]        # 'T'
dna[1:4]      # 'TTC'
dna[:4]       # 'ATTC'
dna[4:]       # 'GGAGCT'
dna[-4:]      # 'AGCT'
dna[1:-1]     # 'TTCGGAGC'
dna[1:-1:2]   # 'TCGG'
dna[::-1]     # 'TCGAGGCTTA'   reverse
