import os


def read_anomalies(file_path):
	anomalies = []
	if os.path.exists(file_path):
		with open(file_path, 'r') as file:
			anomalies = [line.strip() for line in file.readlines()]
	return anomalies


def evaluate_anomaly_detection(dataset_size, groundtruth_anomalies, predicted_anomalies):
	"""
	This function computes TP, FP, TN, FN, Precision, Recall, and F1-score.
	
	Parameters:
	===========
		dataset_size (int): 			Total number of timestamps in the dataset.
		groundtruth_anomalies (set): 	Set of actual anomaly timestamps.
		predicted_anomalies (set): 		Set of predicted anomaly timestamps.
		
	Returns:
	========
		dict: Metrics including TP, FP, TN, FN, Precision, Recall, and F1-score.
	"""
	# Compute TP, FP, FN
	TP = len(groundtruth_anomalies & predicted_anomalies)  # Correctly predicted anomalies
	FP = len(predicted_anomalies - groundtruth_anomalies)  # Incorrectly predicted anomalies (false alarms)
	FN = len(groundtruth_anomalies - predicted_anomalies)  # Missed anomalies

	# Compute Precision, Recall, and F1-score
	precision = TP / (TP + FP) if (TP + FP) > 0 else 0
	recall = TP / (TP + FN) if (TP + FN) > 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	return {
		"TP": TP, "FP": FP, "FN": FN,
		"Precision": precision,
		"Recall": recall,
		"F1-score": f1_score
	}


if __name__ == "__main__":
	groundtruth_file = "../dataset/labels.txt"
	predicted_file = "../dataset/predictions.txt"

	groundtruth_anomalies = read_anomalies(groundtruth_file)
	predicted_anomalies = read_anomalies(predicted_file)

	dataset_size = 4316
	
	metrics = evaluate_anomaly_detection(dataset_size, set(groundtruth_anomalies), set(predicted_anomalies))
	print(metrics)


