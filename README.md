# RoLA Implementation for Real-Time Anomaly Detection

## Overview
This repository contains an implementation of the **RoLA** (Real-time Online Learning Algorithm) for anomaly detection in multivariate time-series sensor data. The implementation follows the methodology described in the RoLA research paper (https://arxiv.org/pdf/2305.16509) and uses **Kafka** for distributing the data and **InfluxDB** for storing the multivariate time series data.
The algorithm detects anomalies based on local dynamic adaptation (LDA) and correlation-based polling to refine anomaly detection results.

This repository contains all code used in the implementation, including:
- Models (RePAD2_modified, RoLA, and LSTM).
- Dataset (multivariate) with labels.
- Docker-infrastructure.
- Scripts for:
    - Sending of data to InfluxDB.
    - Method for evaluating the models.

**Important:**
This implementation is under continuous improvement with the aim of optimizing the RoLA algorithm for better lightweight and efficiency. 

## Prerequisites
To run the code, you need to have the following installed:

- **Anaconda** or other IDEs for managing the Python environment and dependencies.
- **Windows System for Linux (WSL)** to operate Linux under Windows systems.
- **Docker Desktop** or Linux Upuntu for containerizing the Kafka and InfluxDB services.
- **Python 3.8+** and necessary libraries such as `torch`, `influxdb-client`, `pandas`, and `scipy`.

### Steps to Set Up

1. **Clone the Repository**
Clone this repository to your local machine:
```bash
    git clone https://github.com/DirSwe-Art/RoLA_DirSweArt.git
    cd RoLA_DirSweArt
```
2. **Set Up the Environment Create a new Anaconda environment:**
```bash
    conda create --name RoLA-env python=3.8
    conda activate RoLA-env
```
3. **Install Dependencies Install the required libraries by running:**
```bash
    conda install -r requirements.txt
```
4. **Set Up Kafka and InfluxDB Using Docker Compose** This project uses Docker to set up Kafka and InfluxDB services. To start these services, navigate to the **Kafka-TIG** folder from PowerShell command line and run the following commands:
```bash
    wsl -d Ubuntu
    cd ~/Kafka-TIG
    docker-compose -f docker-compose.yml --env-file conf/variables.env up -d
```
   This will spin up the services needed for real-time data distribution and storage.

5. **Send data to InfluxDB** Once the services are up and running, open **Anaconda** command line, navigate to the **scripts** folder, and execute the **send_multivariate_dataset_to_influxdb.py**
6. **Run the RoLA Algorithm** Open another **Anaconda** command line, navigate to **models** folder, and execute the **RoLA_model.py** or you can open **RoLA_model.ipynb** from **Jupiter Notebook**.

7. **Shutdows Kafka** Once you finish testing the algorithm, you can shut down Kafka services by executing the following command in the PowerShell window.
```bash
    docker-compose -f docker-compose.yml --env-file conf/variables.env down
```

*  Note: The "send_multivariate_dataset_to_influxdb.py" is a Python script that sends the data points from the specified CSV file of format 'timestamp', 'value', 'value', 'value', 'value', ....., 'value'  (Check the file multivariate_dataset.csv for reference) to InfluxDB. Screenshots can be found in the **Kafka** folder.
