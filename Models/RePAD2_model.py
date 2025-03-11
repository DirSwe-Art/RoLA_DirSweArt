"""
This file containes a modified copy of RePAD2_model.py taken from miura Github repository
In this version, we do not concider using the InfluxDB, Kafka, and all related tools
"""

# Authors: Dirar Sweidan
# License: 


from RePAD2_LSTM_model import LSTM
import numpy as np
import random
import torch
import time
from datetime import timedelta
from collections import deque

# Setting random seed for reproducibility
torch.manual_seed(140)
np.random.seed(140)
random.seed(140)

# Functions for RePAD2
def calculate_aare(actual, predicted):
	"""
	Calculate the Absolute Relative Error (ARE) between an actual and predicted value.
	
	Parameters:
	actual (deque): The actual value.
	predicted (deque): The predicted value.
	
	Returns:
	float: The Absolute Relative Error.
	"""
	# Adding a small value epsilon to avoid division by zero
	#epsilon = 1e-10
	aare_values = []

	for act, pred in zip(actual, predicted):
		AARE = abs(act - pred) / max(abs(act), 1)
		aare_values.append(AARE)

	mean_aare = np.mean(aare_values)

	return mean_aare

def calculate_threshold(aare_values):
	"""
	Calculate the threshold value (Thd) based on a deque of AARE values.
	Thd is defined as the mean of the AARE values plus three times their standard deviation.

	Parameters:
	- aare_values (array-like): An array of AARE values.

	Returns:
	- float: The calculated threshold value (Thd).
	"""
	# Calculate the mean and standard deviation of the AARE values
	mean_aare = np.mean(aare_values)
	std_aare = np.std(aare_values)
	
	# Calculate Thd
	thd = mean_aare + 3 * std_aare
	
	return thd

# Function for creating and training model
def train_model(train_events):
	tensor_y = torch.tensor(train_events, dtype=torch.float32).view(-1, 1, 1)
	tensor_x = torch.tensor([1, 2, 3], dtype=torch.float32).view(-1, 1, 1)
	# Create an instance of the LSTM model
	model = LSTM(tensor_x, tensor_y, input_size=1, hidden_size=10, num_layers=1, output_size=1, num_epochs=50, learning_rate=0.005)
	
	model.train_model() # Train the model

	return model

# Function for reporting anomalies to InfluxDB (original)
def report_anomaly(T, timestamp, actual_value, predicted_value, write_api):
	"""
	Sends an anomalous event back to InfluxDB, storing it in the "anomaly" measurement
	with both the same value and time as the original event.

	Parameters:
	- anomalous_event: The event data that was detected as an anomaly, including its value and timestamp.
	"""

	point = Point("base_detection-C53")\
		.tag("host", "detector")\
		.field("T", float(T))\
		.field("actual_value", float(actual_value))\
		.field("predicted_value", float(predicted_value))\
		.time(timestamp, WritePrecision.NS)
	
	#write_api.write(bucket="anomalies", org="ORG", record=point)
	#print(f"Anomalous event sent to InfluxDB: Value={actual_value}, Time={timestamp}")

def write_result(timestamp, T, actual_value, predicted_value, AARE, Thd, write_api):
	"""
	Sends an anomalous event back to InfluxDB, storing it in the "anomaly" measurement
	with both the same value and time as the original event.

	Parameters:
	- anomalous_event: The event data that was detected as an anomaly, including its value and timestamp.
	"""

	point = Point("base_result-C53")\
		.tag("host", "detector")\
		.field("T", float(T))\
		.field("actual_value", float(actual_value))\
		.field("predicted_value", float(predicted_value))\
		.field("AARE", float(AARE))\
		.field("Thd", float(Thd))\
		.time(timestamp, WritePrecision.NS)
	
	write_api.write(bucket="anomalies", org="ORG", record=point)
	print(f'T: {T}, Real Value: {actual_value}, Prediction Value: {predicted_value}, AARE: {AARE}, Thd: {Thd}')