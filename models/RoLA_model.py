from LSTM_model import LSTM
import numpy as np
import random
import torch
import time
from datetime import timedelta, datetime
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS
from scipy.stats import pearsonr


# Setting random seed for reproducibility
torch.manual_seed(140)
np.random.seed(140)
random.seed(140)


# Functions for RePAD2
# ====================
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

# Function for reporting anomalies to InfluxDB
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
	
	
# Functions for RoLA
# ==================

def to_rfc3339(timestamp_str):
	# This function converts a timestamp to RFC3339 format
	dt = datetime.fromisoformat(timestamp_str)  # Parse input timestamp
	return dt.strftime('%Y-%m-%dT%H:%M:%SZ')	# Convert to RFC3339 format

def to_normal_time(timestamp_str):
	# This function converts a timestamp to %Y-%m-%d %H:%M format
	dt = datetime.fromisoformat(timestamp_str)  # Parse input timestamp
	return dt.strftime('%Y-%m-%d %H:%M')

def get_previous_values(bucket, measurement, timestamp, num_values, org, url, token, username, password):
	"""
	This function queries the last "num_values" of a single "measurement" from and before the "timestamp" from 
	InfluxDB multi-dimensional dataset in order to compute the correlation coefficient.
	If values before the time stamp are less than "num_values" it gets all previous values.

	Parameters:
	===========
	- bucket (str): 		InfluxDB bucket name.
	- measurement (str):	The variable name to extract.
	- timestamp (str): 		The reference timestamp in RFC3339 format (e.g., "2024-03-20T00:00:00Z").
	- num_values (int): 	The number of values (p) to extract.
	- org (str): 			InfluxDB organization name.
	- url (str): 			InfluxDB server URL.
	- token (str): 			Authentication token.
	- username (str):		Authentication user name.
	- password (str):		Authentication password.

	Returns:
	- List of extracted values for the given variable from and before the timestamp.
	"""
	
	client = InfluxDBClient(url=url, token=token, org=org, username=username, password=password)
	query_api = client.query_api()
	formatted_timestamp = to_rfc3339(str(timestamp))

	# Construct the Flux query to extract one variable's values from a multi-dimensional dataset
	query = f'''
	from(bucket: "{bucket}")
	  |> range(start:  time(v:"2021-10-28T00:00:00Z")) // the earliest timestamp
	  |> filter(fn: (r) => r["_field"] == "{measurement}")
	  |> filter(fn: (r) => r["_time"] <= time(v: "{formatted_timestamp}"))  // Before the timestamp
	  |> sort(columns: ["_time"], desc: true)
	  |> limit(n: {num_values})  // Extract up to num_values
		'''

	# Execute query
	results = query_api.query(query=query, org=org)

	# Extract values
	values = [record.get_value() for table in results for record in table.records]

	#print(f"Extracted values of '{measurement}' before {timestamp}:")
	return values


def fetch_previous_values(var, timestamp, num_values):
	"""Fetches historical values for a given variable."""
	try:
		return get_previous_values(
			bucket=bucket, measurement=var, timestamp=timestamp, num_values=num_values,
			org=org, url=influxdb_url, token=token, username=username, password=password
		)
	except Exception as e:
		print(f"Error fetching values for {var}: {e}")
		return []

def fetch_events():
	"""
	 This function fetches time-series events from InfluxDB.
	 It is used in RoLA to extract the datapoint's relevant variable values from InFluxDB.

	 Returns:
	 ========
	 List of available data points from InfluxDB stored multi-dimensional dataset.
	 """
	query = f'''
		from(bucket: "{bucket}")
		|> range(start: time(v: "{start_time}"))
		|> filter(fn: (r) => r["_measurement"] == "{measurement}")
		|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
	'''
	try:
		return list(query_api.query_stream(org=org, query=query))
	except Exception as e:
		print(f"Error querying InfluxDB: {e}")
		return []
	  
def is_anomaly(T, variable_name, state):
	"""
	This function is an LDA-based anomaly detection function. It checks if a given data point (variable Vx at time T) is an anomaly. 
	It updates a given variable LDA's parameters dynamically.
	In the multivariate case, each flux event consists of a time stamp and a combination of values (variables).
	These values are treated as floats or other data types. Thus, get_value() is not used as in a single-value flux event. 
	
	Parameters:
	===========
	- T (int):				The time point of the current data point.
	- variable_name (str):	The name of the variable used for anomaly detection. 
	- state (dict):			A nested dictionary contains dictionaries (LDAs relevant arguments) associated with each variable. 
							Each dictionary contains specific arguments for individual LDAs to store and update relevant variables data, such as:
	
	* batch_events (deque): A batch of **four time points** events D_T-3, D_T-2, D_T-1, and D_T, that are updated in each iteration.
							It is used for predicting D_T+1 using batch_events[1:], and predicting D_T using batch_events[0:-1].	
	* next_event (deque):	The event to predict next when T = 0, 1, 2, 3, 4, 5, and 6, which is updated in each iteration.
	* M (object):			A trained LSTM model relevant to the current variable. The default value is "None". 
	* flag (bool):			A flag that indicates whether an anomaly was detected (falg=False) in the previous iteration. The default value is "True".
	* actual_value (deque):	A sliding window of three elements to store the actual value of events within three iterations to calculate the AARE.
	* predicted_value (deque):	A sliding window of three elements to store the predicted value of events within three iterations to calculate the AARE.
	* sliding_window_AARE (deque): A sliding window used for storing the AARE resulted in each iteration in order to calculate the threshold later.
	
	Returns:	
	========	
	Anomaly detection Bool value.
	"""

	# Extract state variables
	variable_state = state[variable_name]
	batch_events = variable_state["batch_events"]
	next_event = variable_state["next_event"]
	M = variable_state["M"]
	flag = variable_state["flag"]
	actual_value = variable_state["actual_value"]
	predicted_value = variable_state["predicted_value"]
	sliding_window_AARE = variable_state["sliding_window_AARE"]

	if T < 2:
		return False	# First events, no predictions. No anomaly detections.

	# Initialize LDA
	if 2 <= T < 5:	# Make predictions of D_T+1 by training M with D_T-2, D_T-1, and D_T, i.e., (batch_events)[1:]. 
		if T == 2:	 
			M = train_model(list(batch_events))	   # batch_events contains only 3 values when T=2.
		else:
			M = train_model(list(batch_events)[1:])   # Ignore D_T-3 from the batch_events
		
		variable_state["M"] = M
		pred_D_T_plus_1 = M.predict_next()

		actual_value.append(next_event)
		predicted_value.append(pred_D_T_plus_1)

		return False   # Predictions without AAREs available. No anomaly detections.

	elif 5 <= T < 7:   # Calculate AARE and append to sliding window.
		AARE_T = calculate_aare(actual_value, predicted_value)  # Calculate AARE 
		sliding_window_AARE.append(AARE_T)					  # Uppend to sliding window.

		M = train_model(list(batch_events)[1:])				 # Train M with (D_T-2, D_T-1, and D_T) to predict D_T+1.
		pred_D_T_plus_1 = M.predict_next()

		actual_value.append(next_event)						 # Append the event and its prediction to the sliding window.
		predicted_value.append(pred_D_T_plus_1)

		variable_state["M"] = M
		return False	# Predictions without AARE thresholds available. No anomaly detections.

	elif T >= 7:	 # Make predictions of D_T by training M (when is needed) with D_T-3, D_T-2, D_T-1, i.e., (batch_events)[0:-1]
		if flag:	 # True
			if T != 7:	 # Use previously trained model when T > 7, otherwise, calculate the AARE and the threshold and evaluate the AARE.
				pred_D_T = M.predict_next()		   
				actual_value.append(batch_events[-1])
				predicted_value.append(pred_D_T)

			AARE_T = calculate_aare(actual_value, predicted_value)
			sliding_window_AARE.append(AARE_T)

			# Calculate the threshold only once
			Thd = calculate_threshold(sliding_window_AARE)

			# train a new model if the AARE is larger than Thd, predict, and evaluate again
			if AARE_T > Thd:
				model = train_model(list(batch_events)[0:-1])
				pred_D_T = model.predict_next()

				actual_value.append(batch_events[-1])
				predicted_value.append(pred_D_T)

				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)

				Thd = calculate_threshold(sliding_window_AARE)

				if AARE_T > Thd:	 # anomaly detected
					flag = False
				else:				# AARE <= Thd. No anomaly. Replace the model
					variable_state["M"] = model
					flag = True
		else:   # if the flag is False, train a new model, predict and evaluate AARE
			model = train_model(list(batch_events)[0:-1])
			pred_D_T = model.predict_next()
			actual_value.append(batch_events[-1])
			predicted_value.append(pred_D_T)

			AARE_T = calculate_aare(actual_value, predicted_value)
			sliding_window_AARE.append(AARE_T)

			Thd = calculate_threshold(sliding_window_AARE)

			if AARE_T > Thd:   # anomaly detected
				flag = False
			else:			  # AARE <= Thd. No anomaly. Replace the model
				variable_state["M"] = model
				flag = True

		return not flag


### RoLA Algorithm ###
'''
Implementation:
===============
The algorithm is implemented according to the research RoLA paper (https://arxiv.org/pdf/2305.
 16509). In short, the algorithm continuously streams time-series data from InfluxDB, detects anomalies
for each variable at each time point T, and then, it applies polling based on correlation
analysis to validate the detected anomalies. 

In the beginning, it connects to InfluxDB to stream data and defines parameters such as 
polling interval, correlation thresholds, and sliding window size, then it maintains 
a state dictionary for each variable to store relevant values, models, and flags.

It uses a Flux query to fetch the latest time-series data from InfluxDB. Convert the streamed
data into a list of events. Iterate Over Data Points in the Stream. For each time point T, extract
an event (data point). For each variable in the event, it applies anomaly detection using the 
(is_anomaly function). Store detected anomalies in the list (anomalies) (which resets for each new time point). 
Correlation analysis and polling process are then initialized. If anomalies exist at T, compute 
Pearson’s correlation coefficient between each anomalous variable and all other variables. 
If a strongly correlated variable is also detected as an anomaly, increase the agreement counter. 
Otherwise, increase the disagreement counter. If the number of agreeing anomalies is greater than 
disagreeing ones, the detected anomalies are considered valid and the timestamp is outputted.

To-Do:
======
Implementing parallel processing using PyTorch's parallelism features for running the is_anomaly 
function on all variables at time point T, by utilizing torch.multiprocessing. 
A modification is needed for the is_anomaly function including:

- Using torch.multiprocessing to create a pool of worker processes.
- create a function that prepares tasks for each variable and runs them in parallel using pool.starmap.
- Collect the results (anomalies) and process them in the main loop.

'''

# Setting up the InfluxDB to consume data
influxdb_url = "http://localhost:8086"
token = "random_token"
username = "influx-admin"
password = "ThisIsNotThePasswordYouAreLookingFor"
org = "ORG"
bucket = "system_state"
measurement = "multivariate_dataset"

# Instantiate InfluxDB client
client = InfluxDBClient(url=influxdb_url, token=token, org=org, username=username, password=password)
write_api = client.write_api()
query_api = client.query_api()

# Time series parameters
T = 0
p = 2880  # Number of previous values for correlation
thd_pos, thd_neg = 0.95, -0.95
poll_interval = 1  # Polling frequency in seconds
time_increment = 1  # Time step increment
start_time = "2021-10-28T00:00:00Z"

# Variable names
variables = [
	"SEB45Salinity", "SEB45Conductivity", "OptodeConcentration", "OptodeSaturation",
	"C3Temperature", "FlowTemperature", "OptodeTemperature", "C3Turbidity", "FlowFlow"
]

# State initialization for each variable
state = {
	var: {
		"batch_events": deque(maxlen=4),
		"next_event": deque(maxlen=1),
		"actual_value": deque([0] * 3, maxlen=3),
		"predicted_value": deque([0] * 3, maxlen=3),
		"sliding_window_AARE": deque(maxlen=8064),
		"M": None,
		"flag": True
	}
	for var in variables
}

# List to store processing times
processing_times = []
anomaly_timestamps = []
#anomaly_variable_sets = []
#anomaly_datapoints_sets = []

while True:

	events = fetch_events()  # extract the available data points from a multi-dimensional dataset
	if len(events) < 3:
		print("No sufficient events found.")
		time.sleep(poll_interval)
		continue

	timestamp = 0

	for i, event in enumerate(events):	  # Iterate over each data point.
		iteration_start_time = time.time()  # Start time of this iteration.
		anomalies = set()				   # Reset anomalies for each data point. It is denoted by A in the paper
		timestamp = event["_time"]		  # Extract timestamp.

		# Process each variable in the event
		for var, value in event.values.items():
			if var in ["result", "table", "_start", "_stop", "_time", "_measurement", "host"]:
				continue					# Exclude some metadata.

			# Continuously set batch events.
			state[var]["batch_events"].append(value)

			# Set next event for early iterations (0 ≤ T < 7)
			if i < 7:
				state[var]["next_event"] = events[i + 1][var]

			# Run anomaly detection for the current variable
			if is_anomaly(T, var, state):
				anomalies.add(var)

		# Correlation & Polling Process
		if anomalies:	 # Prepare a p-length window of previous values for each anomalous variable. Store them in a dictionary
			correlated_vars = {a: fetch_previous_values(a, timestamp, p) for a in anomalies}

			for a in anomalies:  # "anomalies" is denoted by list A in the paper 
				C_agree, C_disagree = 1, 0
				L_var, L_data = [a], [event[a]]

				for b in variables:  # b is denoted by the y-th variable in the paper
					if b == a:	   # Ignore b to mitigate redundant processing 
						continue

					a_values = correlated_vars.get(a, [])			 # Fetch the values associated with key "a" in correlated_vars dictionary
					b_values = fetch_previous_values(b, timestamp, p) # Fetch b previous values <= p 

					if not a_values or not b_values:				  # Try just when they are not empty
						continue

					try:
						E_ab, _ = pearsonr(a_values, b_values)
					except Exception:
						continue  # Computation errors

					if thd_neg <= E_ab <= thd_pos:					 # Continue if abs(E_ab) >= thd_pos  
						continue

					if b in anomalies:								 # Polling process
						C_agree += 1
						L_data.append(fetch_previous_values(b, timestamp, 1)) # include the variable value
						L_var.append(b)									   # include the variable name
					else:
						C_disagree += 1

				if C_agree > C_disagree and (C_agree + C_disagree) > 1:
					anomaly_timestamps.append(to_normal_time(str(timestamp)))  # Output results for testing
					#anomaly_variable_sets.append(L_var)
					#anomaly_datapoints_sets.append(L_data)
					#print(f"====> T: {T} ===>  {timestamp}\nVariables: {L_var}\nData: {L_data}")

		# Increment T for the next data point
		T += 1
		iteration_end_time = time.time()

		# Compute and store processing time

		processing_time = iteration_end_time - iteration_start_time
		processing_times.append(processing_time)

		#print(f"Processed timestamp: {timestamp}, Processing time: {processing_time:.4f} seconds")

	# Compute mean and standard deviation every 10 iterations
	#if len(processing_times) % 10 == 0:
	mean_time = np.mean(processing_times)
	std_time = np.std(processing_times)
	print(f"\n\nMean Processing Time of the Received Events: {mean_time:.4f} sec, Std Dev: {std_time:.4f} sec. \n\nTimestamp(s) in which anomalies are detected.")
	for tmstmp in set(anomaly_timestamps):
		print(tmstmp)
		
	# Update start time for the next iteration
	start_time = (timestamp + timedelta(seconds=time_increment)).isoformat()

	time.sleep(poll_interval)