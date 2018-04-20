import os
import csv
import time
from datetime import datetime
import numpy as np

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/features/")

dataset = [0]
dataset_header = ['timestamp']
dataset_time = []

for root, dirs, files in os.walk(dir_path):
	for file in files:
		file_name = file.split(".")[0]
		dataset_header.append(file_name)
		file_path = os.path.join(dir_path, file)
		with open(file_path) as f:
			read_file = csv.reader(f)
			temp_file = []
			temp_timestamp = []
			for row in read_file:
				# Convert to UNIX timestamp
				dt_obj = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple()
				timestamp = int(time.mktime(dt_obj))
				# Check if timestamp between August 2010 until October 2017 in GMT+7
				if timestamp > 1280509200 and timestamp < 1509469200:
					temp_file.append(row[1])
					temp_timestamp.append(row[0])
			dataset[0] = temp_timestamp
			dataset.append(temp_file)

dataset = np.array(dataset).transpose()

myFile = open('dataset.csv', 'w', newline='')

with myFile:
    writer = csv.writer(myFile)
    writer.writerow(dataset_header)
    writer.writerows(dataset)

dataset_norm = dataset[:, 1:].astype(np.float)
# dataset_norm = (dataset_norm - dataset_norm.mean()) / (dataset_norm.max() - dataset_norm.min())
dataset_norm = (dataset_norm - dataset_norm.mean()) / dataset_norm.std()
dataset_norm = dataset_norm.tolist()
for idx, val in enumerate(dataset_norm):
	val.insert(0, dataset[:, 0][idx])
myFileNorm = open('dataset-normalize.csv', 'w', newline='')
with myFileNorm:
	writer = csv.writer(myFileNorm)
	writer.writerow(dataset_header)
	writer.writerows(dataset_norm)