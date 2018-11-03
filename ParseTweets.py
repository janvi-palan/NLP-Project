import os
import sys

#num_lines = sum(1 for lin in open("USER_TWEETS.txt"))
#print(num_lines)
# file is 39,532,926 lines longs (contains that many tweets)

with open("USER_TWEETS.txt") as inp:
    current_line = 0
    for line in inp:
        values    = line.split("\t") # id, ?, tweet
        user_id   = values[0]
        tweet     = values[2]
        file_name = "users/" + user_id
        with open(file_name, "a+") as write_to_file:
            write_to_file.write(tweet + '\r')
        current_line = current_line + 1
        if (current_line % 100000 == 0):
            print('on line: ' + str(current_line))
