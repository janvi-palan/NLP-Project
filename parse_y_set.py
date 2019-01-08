import os


f_output = open("./polarity_oc.txt")
for line in f_output:
	words = line.split()
	print (words[1])