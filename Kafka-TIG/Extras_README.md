Notes:

Repository that was used as a starting point for our implementation:
[/eternalmit5/Kafka-TIG](https://github.com/eternalamit5/Kafka-TIG)

To connect Grafa to InfluxDB, you need to add a data source in Grafana.
Query language: Flux

Find the IP-address of the InfluxDB container by inspecting the network
of the container


Initialized in the `variables.env` file

```bash
InfluxDB Username
InfluxDB Password
InfluxDB ORG_name
InfluxDB Bucket_name
InfluxDB Token
```

'send_multivariate_dataset_to_influxdb' is a Python script that sends the datapoints from the specified csv file. 
of format 'timestamp', 'value', 'value', 'value', 'value', ....., 'value'  (Check the file multivariate_dataset.csv for reference) to InfluxDB.


