import sys
import os
import math
import operator
import random
import matplotlib as mp
import networkx as nx
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict


eCon = "EXTREME_CONSERVATIVE" # 2.5 - 1.5
mCon = "MODERATE_CONSERVATIVE" #1.5 - 0.5
neut = "NEUTRAL" # 0.5 - -0.5
mLib = "MODERATE_LIBERAL" #-0.5 - -1.5
eLib = "EXTREME_LIBERAL" #-1.5 - -2.5
ftot = "TOTAL_FRIENDS"

def computeFraction(userId):
	polarityUser = userPolarityTrain[userId]

	if abs(polarityUser) < 0.5:
		graph[userId]["sameClass"] = len(graph[userId][neut])
		graph[userId]["diffClass"] = len(graph[userId][eCon]) + len(graph[userId][eLib]) + len(graph[userId][mCon]) +len(graph[userId][mLib])
	elif polarityUser > 0.5:
		graph[userId]["sameClass"] = len(graph[userId][mCon]) + len(graph[userId][eCon])
		graph[userId]["diffClass"] = len(graph[userId][eLib]) + len(graph[userId][mLib])
	elif polarityUser < -0.5:
		graph[userId]["diffClass"] = len(graph[userId][mCon]) + len(graph[userId][eCon])
		graph[userId]["sameClass"] = len(graph[userId][eLib]) + len(graph[userId][mLib])

	graph[userId]["diffFrac"] = 	(graph[userId]["sameClass"] - graph[userId]["diffClass"])/float(len(graph[userId][ftot]))


	
def getProcessList():
	file1 = open("out.txt","r")
	for line in file1:
		processList.append(line[:-1])
	file1.close()

def userKeys():
	file = open("polarity.txt","r")
	for line in file:
		fields = line.split()
		u_id = fields[0]
		polarity = float(fields[1])
		if polarity >= -2.5 and polarity < -1.5 :
			polarized_classes[eLib].append(u_id)
		elif polarity >= -1.5 and polarity < -0.5 :
			polarized_classes[mLib].append(u_id)
		elif polarity >= -0.5 and polarity < 0.5 :
			polarized_classes[neut].append(u_id)
		elif polarity >= 0.5 and polarity < 1.5:
			polarized_classes[mCon].append(u_id)
		elif polarity >= 1.5 and polarity < 2.5:
			polarized_classes[eCon].append(u_id)

		userPolarityTrain[u_id] = polarity
	file.close()

def plotDistClasses():
	objects = []
	values = []
	for id in polarized_classes:
		print id
		objects.append(id)
		values.append(len(polarized_classes[id]))

	y_pos = np.arange(len(objects))
	plt.barh(y_pos, values, align='center', color='b')
	plt.yticks(y_pos, objects, fontsize=5)
	plt.xlabel('Number Of Users')
	plt.title('Distribution in Classes')
	plt.show()

def plotDistEC( liberals, conservatives):
	objects = ["Extreme Liberals", "Extreme Conservatives"]
	values = []
	values.append(len(liberals))
	values.append(len(conservatives))
	y_pos = np.arange(len(objects))
	plt.bar(y_pos, values, align='center', color='b')
	plt.yticks(y_pos, objects, fontsize=5)
	plt.xlabel('Number Of Users in Echo Chambers')
	plt.show()




def echoChamberDetectionPartisan():
	echoChamberUsersListLiberals =[]
	echoChambersUsersListConservatives = []
	echoChamber = []
	nonechoChamber = []
	for id in polarized_classes[eCon]:
		if id in graph:
			if graph[id]["diffFrac"] >= 0.80:
				echoChamber.append(graph[id]["diffFrac"])
				echoChambersUsersListConservatives.append(id)
			else:
				nonechoChamber.append(graph[id]["diffFrac"])


	for id in polarized_classes[eLib]:
		if id in graph:
			if graph[id]["diffFrac"] >= 0.80:
				echoChamber.append(graph[id]["diffFrac"])
				echoChamberUsersListLiberals.append(id)
			else:
				nonechoChamber.append(graph[id]["diffFrac"])

	file = open("echoLiberal", "w")
	for id in echoChamberUsersListLiberals:
		file.write(id+"\n")
	file.close()

	file = open("echoConservative", "w")
	for id in echoChambersUsersListConservatives:
		file.write(id+"\n")
	file.close()

	fractionList = []
	for id in polarized_classes[eLib]:
		if id in graph:
				fractionList.append(graph[id]["diffFrac"])
	for id in polarized_classes[eCon]:
		if id in graph:
			fractionList.append(graph[id]["diffFrac"])

	n, bins, patches = plt.hist(fractionList)
	plt.xlabel('Homophily Score : Partisan Users')
	plt.ylabel('Num of Partisan Users ')
	plt.grid(True)
	plt.show()

	plotDistEC(echoChamberUsersListLiberals, echoChambersUsersListConservatives)

def echoChamberDetectionBiPartisan():

	fractionList = []
	for id in polarized_classes[neut]:
		if id in graph:
			fractionList.append(graph[id]["diffFrac"])

	n, bins, patches = plt.hist(fractionList)
	plt.xlabel('Homophily Score : Bipartisan')
	plt.ylabel('Num of BiPartisan Users ')
	plt.grid(True)
	plt.show()

def clusteringCoeff():
	gateKeepers = []
	notGateKeepers = []
	G = nx.DiGraph()
	# print "num nodes in graph map" ,len(graph)
	for id in graph:
		if id not in G:
			G.add_node(id)
	# print "no edges num G", G.number_of_nodes()
	for id in graph:
		for friend in graph[id][ftot]:
			if friend in G:
				G.add_edge(id,friend)
	#
	pr = nx.clustering(G)

	pageRankGateKeepers = []
	pageRankNotGateKeepers = []
	for id in gateKeepers:
		pageRankGateKeepers.append(pr[id])

	for id in notGateKeepers:
		pageRankNotGateKeepers.append(pr[id])

	n, bins, patches = plt.hist(pageRankGateKeepers)
	plt.xlabel('page rank gatekeepers')
	plt.ylabel('Num users ')
	plt.title(r'$\mathrm{Histogram\ of\ fraction}\ \mu=100,\ \sigma=15$')
	#plt.axis([40, 160, 0, 0.03])
	plt.grid(True)
	plt.show()

	n, bins, patches = plt.hist(pageRankNotGateKeepers)
	plt.xlabel('page rank Notgatekeepers')
	plt.ylabel('Num users ')
	plt.title(r'$\mathrm{Histogram\ of\ fraction}\ \mu=100,\ \sigma=15$')
	#plt.axis([40, 160, 0, 0.03])
	plt.grid(True)
	plt.show()






def echoChamberDetectionMOderate():

	fractionList = []
	for id in polarized_classes[mLib]:
		if id in graph:
			fractionList.append(graph[id]["diffFrac"])
	for id in polarized_classes[mCon]:
		if id in graph:
			fractionList.append(graph[id]["diffFrac"])

	n, bins, patches = plt.hist(fractionList)
	plt.xlabel('Network Cohesion Score')
	plt.ylabel('Num of Users ')
	plt.grid(True)
	plt.show()


def echoChamberDetectionAll():

	fractionList = []
	for class11 in polarized_classes:
		for id in polarized_classes[class11]:
			if id in graph:
				fractionList.append(graph[id]["diffFrac"])

	n, bins, patches = plt.hist(fractionList)
	plt.xlabel('Homophily Score : All users')
	plt.ylabel('Num of Users ')
	plt.grid(True)
	plt.show()


def checkForIntersection(filename, userId):

	if os.path.exists(fileName):
		file1 = open(filename,"r")
		userFriends = []
		for line in file1:
			userFriends.append(line[:-1])
		file1.close()
		for id in userFriends:
			if id in userPolarityTrain	:
				if not userId in graph:
					graph[userId] = {"class" : "null", eCon : [], mCon : [], neut :[],mLib: [], eLib :[], ftot :[], "sameClass" : 0, "diffClass" : 0, "diffFrac":0}
					userPolarity = userPolarityTrain[userId]
					if  userPolarity < -1.5 :
						graph[userId]["class"] = eLib
					elif userPolarity >= -1.5 and userPolarity < -0.5 :
						graph[userId]["class"] = mLib
					elif userPolarity >= -0.5 and userPolarity < 0.5 :
						graph[userId]["class"] = neut
					elif userPolarity >= 0.5 and userPolarity < 1.5:
						graph[userId]["class"] = mCon
					elif userPolarity >= 1.5 :
						graph[userId]["class"] = eCon

				fpol = userPolarityTrain[id]
				if fpol >= -2.5 and fpol < -1.5 :
					graph[userId][eLib].append(id)
				elif fpol >= -1.5 and fpol < -0.5 :
					graph[userId][mLib].append(id)
				elif fpol >= -0.5 and fpol < 0.5 :
					graph[userId][neut].append(id)
				elif fpol >= 0.5 and fpol < 1.5:
					graph[userId][mCon].append(id)
				elif fpol >= 1.5 and fpol < 2.5:
					graph[userId][eCon].append(id)
				graph[userId][ftot].append(id)


graph = {}
polarized_classes = {eCon : [], mCon : [], neut :[],mLib: [], eLib :[]}
userPolarityTrain = {}
processList = []
getProcessList()
userKeys()
for friend in processList:
	fileName = friend+"_friends"
	checkForIntersection(fileName, friend)
print len(graph)

polarized_classes1 = {eCon : [], mCon : [], neut :[],mLib: [], eLib :[]}
for id in graph:
		computeFraction(id)
		polarized_classes1[graph[id]["class"]].append(graph[id]["diffFrac"])

gateKeepers = []
notGateKeepers = []

for id in polarized_classes[mCon]:
	if id in graph:
		if abs(graph[id]["diffFrac"]) <= 0.30:
			gateKeepers.append(id)
		else:
			notGateKeepers.append(id)

for id in polarized_classes[mLib]:
	if id in graph:
		if abs(graph[id]["diffFrac"]) <= 0.30:
			gateKeepers.append(id)
		else:
			notGateKeepers.append(id)

print "mod conservative ", len(polarized_classes[mCon])
print "mod liberal ", len(polarized_classes[mLib])
print "gateKeepers" ,len(gateKeepers)
print "NongateKeepers" ,len(notGateKeepers)


G = nx.DiGraph()

print "num nodes in graph map" ,len(graph)
for id in graph:
	if id not in G:
		G.add_node(id)
print "no edges num G", G.number_of_nodes()



for id in graph:
	for friend in graph[id][ftot]:
		if friend in G:
			G.add_edge(id,friend)
#
pr = nx.clustering(G)

pageRankGateKeepers = []
pageRankNotGateKeepers = []
for id in gateKeepers:
	pageRankGateKeepers.append(pr[id])

for id in notGateKeepers:
	pageRankNotGateKeepers.append(pr[id])

n, bins, patches = plt.hist(pageRankGateKeepers)
plt.xlabel('Clustering Coefficient')
plt.ylabel('Num of users identified as Gatekeepers')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(pageRankNotGateKeepers)
plt.xlabel('Clustering Coefficient')
plt.ylabel('Num of users not identified as Gatekeepers')
plt.grid(True)
plt.show()

# polarized_classes2 = {eCon : [], mCon : [], neut :[],mLib: [], eLib :[]}
#
# for id in graph:
# 	polarized_classes2[graph[id]["class"]].append(pr[id])
# print "extreme liberal " + str(len(polarized_classes2[eLib]))
# print "extreme con " + str(len(polarized_classes2[eCon]))
# print "mod lib " + str(len(polarized_classes2[mLib]))
# print "mod con ", len(polarized_classes2[mCon])
#
# for class2 in polarized_classes2:
# 	#print "polarized_classes1[class1] = ", polarized_classes1[class1]
# 	print "class1 = ", class2
# 	n, bins, patches = plt.hist(polarized_classes2[class2])
# 	plt.xlabel('page rank')
# 	plt.ylabel('Num users '+ class2)
# 	plt.title(r'$\mathrm{Histogram\ of\ fraction}\ \mu=100,\ \sigma=15$')
# 	#plt.axis([40, 160, 0, 0.03])
# 	plt.grid(True)
# 	plt.show()






# plot the distribution of classes


# plotDistClasses()
# echoChamberDetectionPartisan()
# echoChamberDetectionBiPartisan()
# echoChamberDetectionMOderate()
# clusteringCoeff()
# echoChamberDetectionAll()
