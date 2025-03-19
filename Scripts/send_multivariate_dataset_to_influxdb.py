import time
import csv
from kafka import KafkaProducer
from datetime import datetime, timezone


def read_from_file(filename):
	"""
	Read from a text file.

	Parameters:
		- filename (str): The path from which the text file is read.
	
	Returns:
		- list of strings.
	"""
	with open(filename, 'r') as file:
		return [line.strip() for line in file]

def convert_to_line_protocol(timestamp, data_dict, dataset="multivariate_dataset", anomaly_name="anomalies", labels=set()):
	"""
	Convert a timestamp and multiple variables into InfluxDB Line Protocol format.

	Parameters:
		- timestamp (str): The timestamp in "YYYY-MM-DD HH:MM:SS" format.
		- data_dict (dict): A dictionary containing variable names as keys and their values.
		- dataset (str): The name of the measurement (default: "sensor_data").
		- anomaly_name (str): Measurement name for anomalies (default: "anomalies").
		- labels (set): A set of timestamps that are considered anomalies.

	Returns:
		- list of str: A list of formatted line protocol entries.
	"""
	# Convert timestamp to nanoseconds since epoch
	#dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
	dt = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
	timestamp_ns = int(dt.timestamp() * 1e9)  # Convert to nanoseconds

	# Construct the line protocol format for multiple variables
	field_set = ",".join([f"{key}={value}" for key, value in data_dict.items()])
	line = f"{dataset} {field_set} {timestamp_ns}"
	
	lines = [line]

	# If the timestamp is an anomaly, create a duplicate entry under "anomalies"
	if timestamp in labels:
		anomaly_line = f"{anomaly_name} {field_set} {timestamp_ns}"
		lines.append(anomaly_line)

	return lines

def send_csv_data(producer, topic, file_path):
	"""
	Reads a CSV file with multiple variables per timestamp and sends formatted data to Kafka.

	Parameters:
		- producer: Kafka producer instance.
		- topic (str): Kafka topic name.
		- file_path (str): Path to the CSV file.
	"""
	with open(file_path, 'r') as file:
		csv_reader = csv.DictReader(file)  									# Read CSV into dictionary format
		
		for row in csv_reader:
			timestamp = row.pop('TimeStamp')  								# Extract timestamp
			data_dict = {key: float(value) for key, value in row.items()}  		# Convert variables to float
			
			line_protocols = convert_to_line_protocol(timestamp, data_dict) 	 # Convert to Line Protocol
			
			for line_protocol in line_protocols:
				producer.send(topic, value=line_protocol.encode('utf-8'))  	# Send to Kafka
				producer.flush()
				print(f"Sent data: {line_protocol}")

			time.sleep(0)  													# Simulate delay


# Choose the dataset and labels to use
dataset		  = "multivariate_dataset" 					# Name of the measurement in InfluxDB
anomaly_name = "labels_multivariate"						# Name of the measurement for anomalies in InfluxDB
labels		   = read_from_file("../dataset/labels.txt")	# Which labels to use


if __name__ == "__main__":
	bootstrap_servers = "localhost:9093" 				# Kafka broker address
	topic = "sea_water" 									# Topic to send sea water multiple sensor data
	file_path = "../dataset/multivariate_dataset.csv"	# multi-dimenssional time-series dataset
	
	# Create Kafka producer
	producer = KafkaProducer(
		bootstrap_servers=bootstrap_servers,
		value_serializer=lambda v: v, 					# Default serialization
		api_version=(2, 0, 2)
	)

	try:
		send_csv_data(producer, topic, file_path)
	except KeyboardInterrupt:
		print("Keyboard interrupt detected. Exiting...")
	finally:
		print("Closing Kafka producer...")
		producer.close()