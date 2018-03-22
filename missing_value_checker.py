import os
import csv
import time
from datetime import datetime

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset")

for root, dirs, files in os.walk(dir_path):
  for file in files:
    file_path = os.path.join(dir_path, file)
    with open(file_path) as f:
      file_name = file.split(".")[0]
      read_file = csv.reader(f)
      temp_dict = []
      temp_data = []
      for row in read_file:
        dt_obj = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").timetuple()
        timestamp = int(time.mktime(dt_obj))
        if timestamp > 1280509200 and timestamp < 1509469200:
          temp_dict.append({file_name: { 'timestamp':timestamp, 'value':row[1] }})
      for i in range(len(temp_dict)):
        # CHECK NULL
        if(i != len(temp_dict) - 1):
          next_row = temp_dict[i][file_name]['timestamp']
          if temp_dict[i+1][file_name]['timestamp'] - temp_dict[i][file_name]['timestamp'] != 172800:
            print(file_name)
            print(temp_dict[i+1][file_name])
            print
        if next_row == temp_dict[i+1][file_name]['timestamp']:
          temp_data.append(temp_dict[i][file_name]['value'])
      print(temp_data)
