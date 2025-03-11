"""Backup of original functions """

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

### REPAD2 Algorithm ###

"""
THE PLAN

Making the program start from any predefined time, and then continue to fetch data from that time and onwards.
And so it can process older data and then catch up to the present time.

It therefore processes a batch of 3 from the earliest timestamp in the range, and then updates the start time for the next iteration 
by incrementing the start time by a set time to both avoid duplicate events and to eventually catch up to the present time.

The program will run indefinitely, and will continue to fetch data from the InfluxDB and process it in batches of 3 until the program is stopped.
I there are not enough events for a batch, the program will wait for a set time before trying again.
When a batch of three is available it will be processed and the it will again wait for another event to be available.

This way it is both flexible and efficient, and can be easily used to process either data in real-time or historical data


For each batch it follows the algorithm of RePAD2


To-Do:

Store actual values (Store whole event?) with predicted values in sliding window 

"""

# Testing of InfluxDB with LSTM
# Må installeres influxdb-client "pip install influxdb-client"
# https://www.influxdata.com/blog/getting-started-with-python-and-influxdb-v2-0/
influxdb_url = "http://localhost:8086"
token = "random_token"
username = "influx-admin"
password = "ThisIsNotThePasswordYouAreLookingFor"
org = "ORG"
bucket = "system_state"
measurement = "C53"

# Instantiate the QueryAPI
client = InfluxDBClient(url=influxdb_url, token=token, org=org, username=username, password=password)
write_api = client.write_api(write_options=ASYNCHRONOUS)
query_api = client.query_api()

# Sliding window for Threshold
actual_value = deque([0] * 3, maxlen=3)
predicted_value = deque([0] * 3, maxlen=3)
sliding_window_AARE = deque(maxlen=8064)

# Other data structures
batch_events = deque(maxlen=3)
next_event = None

# For printing the values
AARE_T = 0
Thd = 0

# Time parameters
poll_interval = 1  # Second(s)
time_increment = 1 # Second(s)
start_time = "1997-04-10T00:00:00Z"

# RePAD2 specific
T = 0
flag = True
M = None # Model


while True:

	# Construct the Flux query
	query = f'''
	from(bucket: "{bucket}")
	 |> range(start: time(v: "{start_time}"))
	 |> filter(fn: (r) => r["_measurement"] == "{measurement}")
	'''
		
	# Query the data
	events = list(query_api.query_stream(org=org, query=query))

	if len(events) > 1: # Need at least 4 to predict next and compar

		for i in range(len(events)-1):
			batch_events.append(events[i])
			next_event = events[i+1]	 

		# RePAD2 Algorithm

		# Set T to the length of the sliding 
			if T >= 2 and T < 5: # 3 total values when T = 2
				M = train_model([event.get_value() for event in list(batch_events)])
				pred_D_T_plus_1 = M.predict_next()
				# Append the event and its prediction to the sliding window
				actual_value.append(next_event.get_value())
				predicted_value.append(pred_D_T_plus_1)

			elif T >= 5 and T < 7:
				# Calculate AARE and append to sliding window
				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)
				M = train_model([event.get_value() for event in list(batch_events)])
				pred_D_T_plus_1 = M.predict_next()
				# Append the event and its prediction to the sliding window
				actual_value.append(next_event.get_value())
				predicted_value.append(pred_D_T_plus_1)

			elif T >= 7 and flag == True:
				if T != 7: # Use M to precdict D_T
					pred_D_T = M.predict_next()
					# Append the event and its prediction to the sliding window
					actual_value.append(next_event.get_value())
					predicted_value.append(pred_D_T)

				# Calculate AARE and append to sliding window
				AARE_T = calculate_aare(actual_value, predicted_value)
				sliding_window_AARE.append(AARE_T)
				# Calculate Thd
				Thd = calculate_threshold(sliding_window_AARE)
				
				if AARE_T <= Thd: pass # Calculate AARE and append to sliding window
				else:
					# Train an LSTM model with D_T-3, D_T-2, D_T-1
					model = train_model([event.get_value() for event in list(batch_events)])
					# Use the model to predict D_T
					pred_D_T = model.predict_next()
					
					# Append the event and its prediction to the sliding window
					actual_value.append(next_event.get_value())
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
						report_anomaly(T+1, next_event.get_time(), actual_value[-1], predicted_value[-1], write_api)
						# Update flag to False
						flag = False
			
			elif T >= 7 and flag == False:
				# Train an LSTM model with D_T-3, D_T-2, D_T-1
				model = train_model([event.get_value() for event in batch_events])
				# Use the model to predict D_T
				pred_D_T = model.predict_next()
				# Append the event and its prediction to the sliding window
				actual_value.append(next_event.get_value())
				predicted_value.append(pred_D_T)

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
					report_anomaly(T+1, next_event.get_time(), actual_value[-1], predicted_value[-1], write_api)
					# Update flag to False
					flag = False

			
			# Write the results to InfluxDB
			write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 

			# Increment T
			T += 1

		# Update start time for the next iteration
		last_event_time = batch_events[-1].get_time()
		# Increment by 1 second to avoid duplicate events
		start_time = (last_event_time + timedelta(seconds=time_increment)).isoformat()
				

	else:
		print("No events found in range.")

	time.sleep(poll_interval)