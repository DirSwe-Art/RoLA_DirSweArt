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

# RoLA specific
T = 0

# Data structures for training and predicting
batch_events = deque(maxlen=4)					# Events from time points: T-3, T-2, T-1, and T
next_event = deque(maxlen=1)						# Used in the case of predicting D_T+1

# Sliding window for calculating the Threshold
actual_value = deque([0] * 3, maxlen=3)			# To append the actual event value in current iteration 
predicted_value = deque([0] * 3, maxlen=3)		# To append the predicted event value in current iteration 
sliding_window_AARE = deque(maxlen=8064)			# Tp append the resulting AARE in the current iteration

M = None											# Trained LSTM model 
flag = True										# True: no anomaly was ditected in the previous iteration

# Time parameters
poll_interval = 1  								# Second(s)
time_increment = 1 								# Second(s)
start_time = "2021-10-28T00:00:00Z" 				# The time of the first event in the time series data

def individual_LDA(T, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event=None, M=None, flag=True):
	# For printing the values
	AARE_T = 0
	Thd = 0

	# Initialize the LDA
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
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 
		
		return T, M, flag, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event

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
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 

		return T, M, flag, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event
		
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
		
		if AARE_T <= Thd: pass 		# D_T is not reported as anomaly
		else:																
			# Train an LSTM model with D_T-3, D_T-2, D_T-1: list(batch_events)[0:-1]		  
			model = train_model([event.get_value() for event in list(batch_events)[0:-1]]) 
			# Use the model to predict D_T
			pred_D_T = model.predict_next()
			
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
				# D_T reported as anomaly immediately, and update flag to False
				flag = False
				return T, M, flag, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event
				#report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)

		
		# Write the results to InfluxDB
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 
	
	elif T >= 7 and flag == False:
		# Train an LSTM model with D_T-3, D_T-2, D_T-1
		model = train_model([event.get_value() for event in list(batch_events)[0:-1]])
		# Use the model to predict D_T
		pred_D_T = model.predict_next()
		# Append the event and its prediction to the sliding window
		actual_value.append(batch_events[-1].get_value())
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
			# D_T reported as anomaly immediately, and update flag to False
			flag = False
			return T, M, flag, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event

			#report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)

	
		# Write the results to InfluxDB (7.time, 7, 7.value, 
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 
	
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



	if len(events) > 1: 						# Need at least 4 to predict next and compare

		for i in range(len(events)):
			batch_events.append(events[i])
			if i < 7: 
				next_event = events[i+1]	 	# used when  0 <= T < 7 to predict the next event. The 7th event is the last one predicted.

			# RePAD2 Algorithm
			T, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event, M, flag = individual_LDA(T, actual_value, predicted_value, sliding_window_AARE, batch_events, next_event=None, M=None, flag=True)
			# Increment T
			T += 1

		# Update start time for the next iteration
		last_event_time = batch_events[-1].get_time()
		# Increment by 1 second to avoid duplicate events
		start_time = (last_event_time + timedelta(seconds=time_increment)).isoformat()
				

	else:
		print("No events found in range.")

	time.sleep(poll_interval)
