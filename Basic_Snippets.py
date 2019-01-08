filename = "polarity_oc.txt"
count = 0
with open(filename) as inp:
	for line in inp:
		line = line[:-1]
		line = line.split()
		if float(line[1]) < -2.5:
			print (line[1])
			count+=1		
print(count)