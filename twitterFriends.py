import tweepy
import os
import time

# Consumer keys and access tokens, used for OAuth
consumer_key        = ''
consumer_secret     = ''
access_token        = ''
access_token_secret = ''

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication

file = open("polarity_1.txt","r")
userPolarityTrain = []
for line in file:
    fields = line.split()
    u_id = fields[0]
    polarity = fields[1]
    userPolarityTrain.append(u_id)
file.close()

api = tweepy.API(auth)
count=0
while count < len(userPolarityTrain):
	try:
		uId = userPolarityTrain.pop(count)
		count+=1
		print ("Checking friends for uId " + uId)
		friends = api.friends_ids(uId)
		fileName = uId + "_friends"
		file = open(fileName, "w") 
		for friend in friends:
			file.write(str(friend)+"\n")
		file.close()	
	except tweepy.RateLimitError:
		print (" I am going to bed for 15 minutes")
		time.sleep(15 * 60)
		count = count - 1
		print (" I am back")
	except tweepy.TweepError as e:
		print (e.reason)

#api.friends_ids(list(userPolarityTrain)[0])

# def checkForIntersection(filename):
	# file = open("polarity.txt","r")
	# userPolarityTrain = []
	# for line in file:
		# fields = line.split()
		# u_id = fields[0]
		# polarity = fields[1]
		# userPolarityTrain.append(u_id)
		
	# file.close()
	# #print userPolarityTrain

	# file1 = open(filename,"r")
	# userFriends = []
	# for line in file1:
		# #print line[:-1] + " 111"
		# userFriends.append(line[:-1]) 
	# file1.close()
	# print userFriends

	# for id in userFriends:
		# if id in userPolarityTrain:
			# print id + " intersected"

##for uId in userPolarityTrain:
	##friends = api.friends_ids(uId)
##	print len(friends)
	##print len(userPolarityTrain.intersection(set(friends)))
