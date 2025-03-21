from LSTM_model import LSTM
import numpy as np
import random
import torch
import time
from datetime import timedelta
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS


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

    point = Point("base_detection_multivariate_dataset")\
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

    point = Point("base_result_multivariate_dataset")\
        .tag("host", "detector")\
        .field("T", float(T))\
        .field("actual_value", float(actual_value))\
        .field("predicted_value", float(predicted_value))\
        .field("AARE", float(AARE))\
        .field("Thd", float(Thd))\
        .time(timestamp, WritePrecision.NS)
    
    write_api.write(bucket="anomalies", org="ORG", record=point)
    print(f'T: {T}, Real Value: {actual_value}, Prediction Value: {predicted_value}, AARE: {AARE}, Thd: {Thd}')
	
	
### REPAD2 Algorithm ###


"""
This is a modified version of the one found on GitHub. Some changes are made and explained by in-line comments.

"""

# Setting up the InfluxDB to consume data
influxdb_url = "http://localhost:8086"
token = "random_token"
username = "influx-admin"
password = "ThisIsNotThePasswordYouAreLookingFor"
org = "ORG"
bucket = "system_state"
measurement = "multivariate_dataset" 

# Instantiate the QueryAPI
client = InfluxDBClient(url=influxdb_url, token=token, org=org, username=username, password=password)
write_api = client.write_api(write_options=ASYNCHRONOUS)
query_api = client.query_api()

# Sliding window for Threshold
actual_value = deque([0] * 3, maxlen=3)
predicted_value = deque([0] * 3, maxlen=3)
sliding_window_AARE = deque(maxlen=8064)

# Other data structures
batch_events = deque(maxlen=4)								# changed maxlen from 3 to 4 to include: T-3, T-2, T-1, and T
next_event = deque(maxlen=1)								# changed to deque for better processing

# For printing the values
AARE_T = 0
Thd = 0

# Time parameters
poll_interval = 1  # Second(s)
time_increment = 1 # Second(s)
start_time = "2021-10-28T00:00:00Z" 
			 

# RePAD2 specific
T = 0
flag = True
M = None # Model


while True:
	
	# Construct the Flux query
	#query = f'''
	#from(bucket: "{bucket}")
	#	|> range(start: time(v: "{start_time}"))
	#	|> filter(fn: (r) => r["_measurement"] == "{measurement}")
	#'''
	# Query the data
	#events = list(query_api.query_stream(org=org, query=query))

	# Construct the Flux query to retrieve only one variable and timestamp
	query = f'''
	from(bucket: "{bucket}")
		|> range(start: time(v: "{start_time}"))
		|> filter(fn: (r) => r["_measurement"] == "{measurement}")
		|> filter(fn: (r) => r["_field"] == "SEB45Salinity")  
		|> keep(columns: ["_time", "_value"])  
	'''
	# Query the data
	events = list(query_api.query_stream(org=org, query=query))   

	# Construct the Flux query for multiple variables
	#query = f'''
	#		from(bucket: "{bucket}")
	#			|> range(start: time(v: "{start_time}"))
	#			|> filter(fn: (r) => r["_measurement"] == "{measurement}")
	#			|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
	#		'''

	# Query the data
	#events = list(query_api.query_stream(org=org, query=query))



	if len(events) > 1: 						# Need at least 3 to predict next and compare

		for i in range(len(events)):
			batch_events.append(events[i])
			if i < 7: 
				next_event = events[i+1]	 	# used when  0 <= T < 7 to predict the next event. The 7th event is the last one predicted.

			# Print the default outputs when T=0 and T=1
			if i in [0,1]: 
				write_result(batch_events[-1].get_time(), i, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api)
			
			# RePAD2 Algorithm

			# Set T to the length of the sliding 
			if T >= 2 and T < 5:
				# Make predictions of D_T+1 by training M with D_T-2, D_T-1, and D_T, i.e., (batch_events)[1:]. 
				
				# batch_events containes only 3 values when T=2.
				if T==2: 
					M = train_model([event.get_value() for event in list(batch_events)])
				
				# Ignore D_T-3 from the batch_events
				else:
					M = train_model([event.get_value() for event in list(batch_events)[1:]])		 
				
				pred_D_T_plus_1 = M.predict_next()
				
				# Append the event and its prediction to the sliding window.
				actual_value.append(next_event.get_value())
				predicted_value.append(pred_D_T_plus_1)
				
				# Write the results to InfluxDB.
				write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 

			elif T >= 5 and T < 7:
				# Calculate AARE and append to sliding window.
				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)
				
				# Train M with (D_T-2, D_T-1, and D_T) to predict D_T+1.
				M = train_model([event.get_value() for event in list(batch_events)[1:]])
				pred_D_T_plus_1 = M.predict_next()
				
				# Append the event and its prediction to the sliding window.
				actual_value.append(next_event.get_value())
				predicted_value.append(pred_D_T_plus_1)

				# Write the results to InfluxDB.
				write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 

			# Make predictions of D_T by training M with D_T-3, D_T-2, D_T-1, i.e., (batch_events)[0:-1]
			elif T >= 7 and flag == True:							
				if T != 7:
					# Use M to precdict D_T.
					pred_D_T = M.predict_next()
					
					# Append the event (last event in batch_events) and its prediction to the sliding window.
					actual_value.append(batch_events[-1].get_value())
					predicted_value.append(pred_D_T)

				# Calculate AARE and append to sliding window	
				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)
				
				# Calculate Thd
				Thd = calculate_threshold(sliding_window_AARE)
				
				if AARE_T <= Thd: pass # D_T is not reported as anomaly
				else:																
					# Train an LSTM model with D_T-3, D_T-2, D_T-1: list(batch_events)[0:-1]		  
					model = train_model([event.get_value() for event in list(batch_events)[0:-1]]) 
					# Use the model to predict D_T
					pred_D_T = model.predict_next() #; print('==T==> T:', T, 'Batch_events:', batch_events )
					
					# Append the event (last event in batch_events) and its prediction to the sliding window
					actual_value.append(batch_events[-1].get_value())
					predicted_value.append(pred_D_T)

					# Re-calculate AARE_T
					AARE_T = calculate_aare(actual_value, predicted_value)
					sliding_window_AARE.append(AARE_T)
					
					# Re-calculate Thd
					Thd = calculate_threshold(sliding_window_AARE)

					if AARE_T <= Thd:
						# D_T is not reported as anomaly
						# Replace M with the new model
						M = model
						# Update flag to True
						flag = True
					else:
						# D_T reported as anomaly immediately
						report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)
						# Update flag to False
						flag = False
				
				# Write the results to InfluxDB
				write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 
			
			elif T >= 7 and flag == False:
				# Train an LSTM model with D_T-3, D_T-2, D_T-1
				model = train_model([event.get_value() for event in list(batch_events)[0:-1]])
				# Use the model to predict D_T
				pred_D_T = model.predict_next()
				# Append the event and its prediction to the sliding window
				actual_value.append(batch_events[-1].get_value())
				predicted_value.append(pred_D_T) #; print('==F==> T:', T, 'Batch_events:', batch_events )

				# Calculate AARE_T
				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)
				# Calculate Thd
				Thd = calculate_threshold(sliding_window_AARE)

				if AARE_T <= Thd:
					# D_T is not reported as anomaly
					# Replace M with the new model
					M = model
					# Update flag to True
					flag = True
				else:
					# D_T reported as anomaly immediately
					report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)
					# Update flag to False
					flag = False
			
				# Write the results to InfluxDB (7.time, 7, 7.value, 
				write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 

			# Increment T
			T += 1

		# Update start time for the next iteration
		last_event_time = batch_events[-1].get_time()
		# Increment by 1 second to avoid duplicate events
		start_time = (last_event_time + timedelta(seconds=time_increment)).isoformat()
				

	else:
		print("No events found in range.")

	time.sleep(poll_interval)
