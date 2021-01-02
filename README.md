# structuring_code

## 1 What does the data look like?
### 1.1 dataframe_raw.csv
The households were instructed to turn on the smart meters before they started to cook with the EPC and to
turn them off when they were finished with their cooking activities. This meant that most of the meters were
turned off when the households weren’t cooking. The raw data is presented in CSV-format with a recording
frequency of 5 minutes.

<table>
  <tr>
    <th>COLUMN NAME</th>
    <th>VALUE</th>
    <th>DESCRIPTION</th>
  </tr>
  <tr>
    <td>meter_number</td>
    <td>ID</td>
    <td>unique identifier of smart meter in household</td>
  </tr>
  <tr>
    <td>timezone</td>
    <td>UTC+HH:MM</td>
    <td>timezone of timestamp</td>
  </tr>
  <tr>
    <td>timestamp</td>
    <td>YYYY-MM-DD HH:MM:SS</td>
    <td>timestamp at time of measurement</td>
  </tr>
  <tr>
    <td>energy</td>
    <td>Kilowatt hour, kWh</td>
    <td>cumulative energy used by appliance</td>
  </tr>
  <tr>
    <td>voltage</td>
    <td>Volt, V</td>
    <td>active voltage at time of measurement</td>
  </tr>
  <tr>
    <td>current</td>
    <td>Ampere, A</td>
    <td>active current at time of measurement</td>
  </tr>
  <tr>
    <td>power</td>
    <td>Kilowatt, kW</td>
    <td>active real power at time of measurement</td>
  </tr>
  <tr>
    <td>power_factor</td>
    <td>Ratio, -</td>
    <td>apparent power / real power</td>
  </tr>
  <tr>
    <td>frequency</td>
    <td>Herz, Hz</td>
    <td>frequency at time of measurement</td>
  </tr>
</table>

#### 1.1.1 Excluding Outliers
We have excluded specific outliers from the raw data. These outliers are corrupt data, which
appears in the energy column as big data spikes due to SIM-card issues, modem changes and grid
outages. Specifically, all outliers that are resulting in a spike above 1 kWh are corrected. 

![Data spike](/images/546336_comb_spikes.png)

## 2 Processing code

### 2.1 Code Language
We used Python with Pandas for the data processing. Therefore, all code examples are given in
Python syntax.

### 2.2 Defining the Cooking Event
The cooking event is defined as a consistent recording sequence by monitoring the energy
consumption and power level. Several conditions are put up to make the cooking event definitions as
precise as possible (see deep dive). The time between a recording of energy consumption and the
end of a cooking session is set to 15 minutes, which is equal to 3 recordings at 5 minute
recording frequency.

### 2.3 Handling Cooking Event Duplicates
During the pilot, we have been developing a backfilling functionality. It allows us to send stored data
to the server when a meter comes to an area with a good network. Unfortunately, some "baby
diseases" appeared at the beginning of using this feature. One such issue was that cooking events
appeared twice, because smart meters sent data before they had been synchronized to the
timezone settings of the server. Fortunately, this issue is now solved and, thus, only appears in the
first half of the pilot.
The processed dataframe is excluding cooking events duplicates. We are, furthermore, expecting
that the backfilling functionality to reduce data gaps will become significantly better until the final
data release at the end of the pilot.

![Event duplicate](/images/546281_timeissue.png)

### 2.4 Adding Start & Ending of Cooking Events
As the EPCs are turned off when they aren't used, the recordings do not include the time that the EPCs haven't sent their first signal in a cooking event. Similarily, when the cooking is finished the EPC might be turned off before it has sent the last recording. Hence, some of the energy consumption that clearly belongs to a specific cooking event might show up as a gap between events. This energy is added retrospectively by adding the time to each event before the first recording and after the last recording. This time is calculated based on the energy consumption difference and can be up to the time resolution interval, i.e. the smart meters has a 5 minute interval for this EPC project. The result is a more accurate consumption of cooking events and better estimates of the lost cooking events.



### Deep Dive - Cooking Event Algorithm
As mentioned before, Python with Pandas was used for data processing. Below is a description of
the steps that were taken to define the cooking events:
1) Defining new parameters:
<ul>
  <li>power = current * voltage</li>
  <li>min_active_load = 15%</li>
  <li>power_capacity = 1 kW</li>
  <li>time_resolution = 5 minutes</li>
  <li>t_between = 15 minutes</li>
</ul>

2) Create columns based on columns 'meter_number' and 'timestamp' by selecting the time difference between rows for each meter_number to conduct the further analysis:
<ol type="a">
  <li> <code>df_processed.loc[(df_processed.meter_number.diff() == 0),
                     'diff_prev_timestamp'] = df_processed.timestamp.diff()</code> </li>
  <li> <code>df_processed.loc[(df_processed.meter_number.diff(-1) == 0),
                     'diff_next_timestamp'] = df_processed.timestamp.shift(-1) - df_processed.timestamp</code> </li>
</ol>


3) A column called ‘load’ is created to indicate when the EPC is active, i.e. when a power
load is applied:
<ol type="a">
  <li> <code>energy - energy.shift() > min_active_load * power_capacity * time_resolution / 60</code> </li>
  <li> <code>power > min_active_load * power_capacity</code> </li>
</ol>


4) A summary of the start and end conditions of the cooking events are found in the illustration
below.
![Event algorithm](/images/cooking_event_picture_structure.png)
