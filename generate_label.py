import os
import csv
import time
from datetime import datetime
import numpy as np

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/market-price.csv")
dataset = []
original_file = []

with open(file_path) as f:
	read_file = csv.reader(f)
	for row in read_file:
		dt_obj = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple()
		timestamp = int(time.mktime(dt_obj))
		# Check if timestamp after August 2010
		if timestamp > 1280509200:
			original_file.append(row)

for i in range(1, 31):
	file_name = str(i*2) + 'days' + '.csv'
	temp = []
	for j in range(len(original_file)):
		# Convert to UNIX timestamp
		dt_obj = datetime.strptime(original_file[j][0], "%Y-%m-%d %H:%M:%S").timetuple()
		timestamp = int(time.mktime(dt_obj))
		# Check if timestamp between August 2010 until October 2017
		if timestamp > 1280509200 and timestamp < 1509469200:
			label = [1, 0] if float(original_file[j][1]) > float(original_file[j+i][1]) + (float(original_file[j+i][1])*0.01) else [0, 1]
			# print(original_file[j][1], original_file[j+1][1], float(original_file[j+i][1]) + (float(original_file[j+i][1])*0.01), label)
			assert label == [1, 0] if float(original_file[j][1]) > float(original_file[j+i][1]) + (float(original_file[j+i][1])*0.01) else [0, 1]
			temp.append(label)
		dataset.append(temp)
	my_file = open('labels/' + file_name, 'w', newline='')
	with my_file:
		writer = csv.writer(my_file)
		writer.writerows(temp)

# for i in range(0,60,2):
#   file_name =  str(i+2 )+ 'days' + '.csv'
#   with open(file_path) as f:
#     read_file = csv.reader(f)
#     temp_file = []
#     for row in read_file:
#       # Convert to UNIX timestamp
#       dt_obj = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple()
#       timestamp = int(time.mktime(dt_obj))
#       # Check if timestamp between August 2010 until October 2017
#       if timestamp > 1280509200 and timestamp < 1509469200:
#         temp_file.append(row[1])

#       myFile = open(file_name, 'w')
#       with myFile:
#         writer = csv.writer(myFile)
#         writer.writerows([dataset])
