### RoLA Algorithm ###
from scipy.stats import pearsonr

def get_previous_values(bucket, measurement, timestamp, num_values, org, url, token, , username, password):
    """
    This function queries the last "num_values" of a single "measurement" before "timestamp" from 
	InfluxDB multi-dimensional dataset in order to compute the correlation coeffecient.
	If velues before the time stamp are less than "num_values" it gets all previous values.

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
    - List of extracted values for the given variable.
    """

	client = InfluxDBClient(url=url, token=token, org=org, , username=username, password=password))
	query_api = client.query_api()

	# Construct the Flux query to extract one variable's values from a multi-dimensional dataset
	query = f'''
    from(bucket: "{bucket}")
	  |> range(start: -∞)
	  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
	  |> filter(fn: (r) => r["_time"] < time(v: "{timestamp}"))  // Before the timestamp
	  |> sort(columns: ["_time"], desc: true)
	  |> limit(n: {num_values})  // Extract up to num_values
		'''

	# Execute query
	results = query_api.query(query=query, org=org)

	# Extract values
	values = [record.get_value() for table in results for record in table.records]

	#print(f"Extracted values of '{measurement}' before {timestamp}:")
	return values
	
	
	
def is_anomaly(T, variable_name, state):
	"""
    This function is an LDA-based anomaly detection function that checks if a given
	data point (variable Vx at time T) is an anomaly. 
	It updates variable LDA's parameters dynamically.
	In the multivariate case, each flux event consists of a time stamp and a combination of values.
	These values are treated as floats or other data types. Thus, get_value() was not used as we did a flux event with an one value. 
	
	Parameters:
	===========
	- T (int):				The given time point of the data point.
	- variable_name (str):	The name of the variable of the data point. 
	- state (dict):			A nested dictionary containes dictionaries associated with each variable, which containes
							specific arguments for an LDA to store and update relevant data, such as:
	
	* batch_events (deque): A batch of four time points events D_T-3, D_T-2, D_T-1, and D_T. It should be updated in each iteration.
							It is used for predicting D_T+1 using batch_events[1:], and predicting D_T using batch_events[0:-1].	
	* next_event (deque):	The event to predict next when T = 0, 1, 2, 3, 4, 5, and 6. It should be updated in each iteration.
	* M (object):			A trained LSTM model. Default value is "None". 
	* flag (bool):			A flag that indicates whether an anomaly was ditected (falg=False) in the previous iteration. Default value is "True".
	* actual_value (deque):	A deque type window of three elements to store the actual value of events within three iterations to calculate the AARE.
	* predicted_value (deque):	A deque type window of three elements to store the predicted value of events within three iterations to calculate the AARE.
	* sliding_window_AARE (deque): A deque type window to store the AARE resulted in each iteration in order to calculate the threhold later.
	
	Return:	
	=======	
	The flag indicating the anomaly, together with updated batch events, next events, actual_value, predicted_value, 
	sliding_window_AARE, and the model that will be used in the next iteration.
    """
	
	# For printing the values
	AARE_T = 0
	Thd = 0
	variable_state   = state[variable_name]
	
	batch_events	= variable_state["batch_events"]
	next_event    	= variable_state["next_event"]
	M                   	= variable_state["M"]
	flag			= variable_state["flag"]
	actual_value	= variable_state["actual_value"]
	predicted_value	= variable_state["predicted_value"]
	sliding_window_AARE = variable_state["sliding_window_AARE"]

	# Initialize the LDA
	if T >= 2 and T < 5:
		# Make predictions of D_T+1 by training M with D_T-2, D_T-1, and D_T, i.e., (batch_events)[1:]. 
		
		# batch_events containes only 3 values when T=2.
		if T==2: 
			M = train_model([event for event in list(batch_events)])
			variable_state["M"] = M
		
		# Ignore D_T-3 from the batch_events
		else:
			M = train_model([event for event in list(batch_events)[1:]])
			variable_state["M"] = M
		
		pred_D_T_plus_1 = M.predict_next()
		
		# Append the event and its prediction to the sliding window.
		actual_value.append(next_event)
		predicted_value.append(pred_D_T_plus_1)
		
		# Write the results to InfluxDB.
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 
		
		#return batch_events, next_event, M, actual_value, predicted_value, sliding_window_AARE, flag
		return False

	elif T >= 5 and T < 7:
		# Calculate AARE and append to sliding window.
		AARE_T = calculate_aare(actual_value, predicted_value)
		sliding_window_AARE.append(AARE_T)
		
		# Train M with (D_T-2, D_T-1, and D_T) to predict D_T+1.
		M = train_model([event for event in list(batch_events)[1:]])
		pred_D_T_plus_1 = M.predict_next()
		
		# Append the event and its prediction to the sliding window.
		actual_value.append(next_event)
		predicted_value.append(pred_D_T_plus_1)

		# Update M for the next iteration. This has no effect with T=6, but is needed in the iteration when T>7
		variable_state["M"] = M
		
		# Write the results to InfluxDB.
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-2], AARE_T, Thd, write_api) 

		#return batch_events, next_event,  M, actual_value, predicted_value, sliding_window_AARE, flag
		return False
		
	# Make predictions of D_T by training M with D_T-3, D_T-2, D_T-1, i.e., (batch_events)[0:-1]
	elif T >= 7 and flag == True:							
		if T != 7:
			# Use M to precdict D_T.
			pred_D_T = M.predict_next()
			
			# Append the event (last event in batch_events) and its prediction to the sliding window.
			actual_value.append(batch_events[-1])
			predicted_value.append(pred_D_T)

		# Calculate AARE and append to sliding window	
		AARE_T = calculate_aare(actual_value, predicted_value)
		sliding_window_AARE.append(AARE_T)
		
		# Calculate Thd
		Thd = calculate_threshold(sliding_window_AARE)
		
		if AARE_T <= Thd: pass 		# D_T is not reported as anomaly
		else:																
			# Train an LSTM model with D_T-3, D_T-2, D_T-1: list(batch_events)[0:-1]		  
			model = train_model([event for event in list(batch_events)[0:-1]]) 
			# Use the model to predict D_T
			pred_D_T = model.predict_next()
			
			# Append the event (last event in batch_events) and its prediction to the sliding window
			actual_value.append(batch_events[-1])
			predicted_value.append(pred_D_T)

			# Re-calculate AARE_T
			AARE_T = calculate_aare(actual_value, predicted_value)
			sliding_window_AARE.append(AARE_T)
			
			# Re-calculate Thd
			Thd = calculate_threshold(sliding_window_AARE)

			if AARE_T <= Thd:
				# D_T is not reported as anomaly
				# Replace M with the new model
				#M = model
				variable_state["M"] = model
				# Update flag to True
				flag = True
			else:
				# D_T reported as anomaly immediately, and update flag to False
				flag = False
				#report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)
				
		# Write the results to InfluxDB																							### delete later
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 	### delete later
		#return batch_events, next_event, M, actual_value, predicted_value, sliding_window_AARE, flag
		return not flag	

	elif T >= 7 and flag == False:
		# Train an LSTM model with D_T-3, D_T-2, D_T-1
		model = train_model([event for event in list(batch_events)[0:-1]])
		# Use the model to predict D_T
		pred_D_T = model.predict_next()
		# Append the event and its prediction to the sliding window
		actual_value.append(batch_events[-1])
		predicted_value.append(pred_D_T) 

		# Calculate AARE_T
		AARE_T = calculate_aare(actual_value, predicted_value)
		sliding_window_AARE.append(AARE_T)
		# Calculate Thd
		Thd = calculate_threshold(sliding_window_AARE)

		if AARE_T <= Thd:
			# D_T is not reported as anomaly
			# Replace M with the new model
			#M = model
			variable_state["M"] = model
			# Update flag to True
			flag = True
		else:
			# D_T reported as anomaly immediately, and update flag to False
			flag = False
			#report_anomaly(T, batch_events[-1].get_time(), actual_value[-1], predicted_value[-1], write_api)
		
		# Write the results to InfluxDB																							### delete later
		#write_result(batch_events[-1].get_time(), T, batch_events[-1].get_value(), predicted_value[-1], AARE_T, Thd, write_api) 	### delete later
		#return batch_events, next_event, M, actual_value, predicted_value, sliding_window_AARE, flag
		return not flag
		
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

# Sttings
T = 0											# Time point
A, L_var, L_data       = [], [], []					# Lists for storing polling results
C_agree, C_disagree = 0, 0						# Counters
p                               = 2880						# The interval between two time points used in computing the correlation between two variables
thd_pos, thd_neg      = 0.95, -0.95					# The positive and negative Pearson's correlation coefficient thresholds
poll_interval = 1  								# Time parameter: Second(s)
time_increment = 1 								# Time parameter: Second(s)
start_time = "2021-10-28T00:00:00Z" 				# The timestamp of the first event to start with in the given time series dataset


# Initialize a state dictionary for each variable where each LDA stores and updates its arguments, such as 
# the batch events, trained model, the flag, and others.
variables = ["SEB45Salinity", "SEB45Conductivity", "OptodeConcentration", "OptodeSaturation", 
			"C3Temperature", "FlowTemperature", "OptodeTemperature", "C3Turbidity", "FlowFlow"]
state        = { 
			key:{"batch_events": 	deque(maxlen=4), 
				 "next_event": 		deque(maxlen=1), 
				 "actual_value": 	deque([0] * 3, maxlen=3), 
				 "predicted_value": 	deque([0] * 3, maxlen=3), 
				 "sliding_window_AARE": deque(maxlen=8064), 
				 "M": 				None, 
				 "flag": 			True 
				 }
			for key in variables 
			}

   
	
while True:
	# Construct the Flux query to extract the available data points from a multi-dimensional dataset
	query = f'''
			from(bucket: "{bucket}")
				|> range(start: time(v: "{start_time}"))
				|> filter(fn: (r) => r["_measurement"] == "{measurement}")
				|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
			'''

	# Query the data
	events = list(query_api.query_stream(org=org, query=query))

	
	if len(events) > 1: 						# Need at least 3 events to predict next.
		timestamp = 0						# The timestamp of the last processed event.
		for i in range(len(events)):			# Iterate over each data point.
			event = events[i]				# The data point expressed by N-dimensional vector at time point T received by Flux query of readings sent by Kafka1.
			timestamp = event["_time"]  		# Extract timestamp.
				
			# Iterate over each variable. Distribute variables to LDAs
			for variable, value in event.values.items():
				if variable in ["result", "table","_start","_stop","_time","_measurement","host"]: continue

				# Send the value (at time T) to the LDA's "bach_events" argument stored in the state dictionary.
				state[variable]["batch_events"].append(value)
				
				# Send the next value (at time T+1) to the LDA's "next_event" argument stored in the state dictionary. 
				# This is used for the LDA to predict the next event when  0 <= T < 7. The 7th event is the last one predicted.
				if i < 7: 
					state[variable]["next_event"] = events[i+1][variable]	 	

				# Run anomaly detection of the current variable at time T with updated state.
				anomaly = is_anomaly(T, variable, state)
				if anomaly:
					A.append(variable)

			# Calculate the Pearson's correlation coefficients between the anomalous variables and proceed with polling
			if len(A) > 0:
				for y in range(0,len(A)):
					C_agree      	= 1
					C_disagree 	= 0
					L_var   		= []
					L_data 		= []
					a = A[y]			
					L_var.append(a)
					Sa_T = event[a]
					L_data.append(Sa_T)
					
					# Correlations process
					for z in range(len(variables)):
						if variables[z] not a:
							b = variables[z]
							a_values = get_previous_values(bucket=bucket, measurement=a, timestamp=timestamp, num_values=p, org=org, url=influxdb_url, token=token,username=username, password=password)
							b_values = get_previous_values(bucket=bucket, measurement=b, timestamp=timestamp, num_values=p, org=org, url=influxdb_url, token=token,username=username, password=password)
							E_ab, _ =  pearsonr(a_values, b_values)
							
							# Polling process
							if abs(E_ab) >= thd_pos:
								if b in A:
									C_agree += 1
									L_data.append(value)
									L_var.append(b)
								else:
									C_disagree += 1
					if (C_agree > C_disagree) and ( (C_agree + C_disagree) > 1):
						for T_data, T_var in zip(L_data, L_var):
							print(timestamp, T_data, T_var)
			# Reset A
			A = []
			# Increment T
			T += 1

		print('timestamp', timestamp)
		# Update start time for the next iteration
		last_event_time = timestamp
		#last_event_time = batch_events[-1].get_time()
		# Increment by 1 second to avoid duplicate events
		start_time = (last_event_time + timedelta(seconds=time_increment)).isoformat()
				

	else:
		print("No events found in range.")

	time.sleep(poll_interval)
