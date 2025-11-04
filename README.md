## Overview

EMG control for the Prensilia MIA proesthetic hand. The system converts muscle activity from the forarm into control commands that drive the hands 3-motor actuators. The system goes from EMG acquisition and preprossing to an ML gesture classification then sending that over BLE to the hand

For input, the OyMotion GForce+ armband records 8 channels (it has 8 electrodes) of surface EMG at 500 Hz. The armand transmits raw 12-bit samples BLE whuch are received by a Python client using the *bleak* library. This signal stream can be published to the **Lab Streaming Layer (LSL)** for synchronized recording, allowing for precice alignment with gesture cue timestamps during training. The recorded data is later converted from `.xdf` format into MATLAB .mat strucutres for analysis and model training.

The code extracts time-domain EMG features including **mean absolute value**, **variance**, **waveform length**, **zero crossings**, and **slope sign changes**, from overlapig signal windows of 125 samples with 25 sample hops. These features go into a **Linear Discriminant Analysis (LDA)** classifier, which maps EMG patterns to discrete gesture classes such as open or close. This classifier is used for both offline (training/validation) and online (real-time inference), for consistency between data collection and control.

During operation, Python continously receives streaming EMG from the armband, processes each new window through the feature extractor, and generates gesture predictions every 50 ms. These predictions are stabalized using a uniform voting filter to prevent rapid flickering. Once a gesture decision is confirmed, the classifier sends an integer class index through a local TCP socket to a BLE module.

Arduino Nano 33 IoT serves as the BLE bridge between the host computer and the MIA hand. It receives the gesture index over the socket, maps it to an 18-character ASCII command recognized by the MIA firmware, and transmits it through UART. The MIA Hand's onboard contorller can interpret these commands to actuate its motor.

![EMG Robotic Control Demo](hand_control.gif)

## Pipeline

### 1. Signal Acquisition 
Continuously streams multi-channel surface EMG signals from the **OyMotion GForce+** armband, decodes the raw bluetooth packets into calibrated microvolt values, and publish these streams to the LSL for synchronized recording.

This processes is mostly handled by the file **```control/OYMotion.py```** which implements the BLE connection and raw-data unpacking. The function ```connect_armband_and_start_streaming(mac, usingLSL=True)``` uses the *bleak* library to connect to the armband via its MAC address. 

Once connected, it sends configuration commands to the device, selecting a 500 Hz sampling rate, 12-bit resultion, and 8 channels, through the ```EmgRawDataConfig``` class. The device begins to stream binary EMG packets that are receieved as byte arrays.

Each incoming packet is decoded by the private method ```_convert_emg_to_uv(data: bytes)``` which performs the conversion from the devices signed integer representation to physical microvolts. 

The raw samples are divided by the amplifier gain and scaled according to the ADC reference voltage. The function reshapes the resulting NumPy array into an N×8 matrix, where each column corresponds to one electrode channel on the armband. These samples represent instantaneous EMG amplitudes.

When LSL recording is enabled, ```init_lsl()``` in the same module creates an outlet stream wiht the name "OyMotion GForce+" and type "EMG", containing 8 channels sampled at 500 Hz. This stream allows the EMG to be timestamped and synchronized with other sources such as gesture cues. Files like ```train/CollectEMGData.py``` and ```control/CollectEMGData.py``` call this function to push EMG data into the LSL network during both training and runtime.

Overall, the output of this stage is a continious stream of EMG samples represented as an array of floating-point microvolt values and can be recorded to disk or passed directly into the classifier.

### 2. Gesture Cue and Labeling

This part provides ground-truth labels needed to train supervised gesture classifiers. During data collection, it ensures that each segment of EMG data corresponds preciesely to a known gesture. Without this synchronization, the classifier would recieve unlabeled EMG and not be possible to train.

This is implemented in **t```train/ShowCues.py```**, which uses OpenCV to display random gesture prompts to the person while simultaneosuly publishing LSL marker streams that envode the onset and offset of each gesture window. 

The functions ```init_lsl()``` and ```init_lsl_grip_strength()``` create 2 LSL outlet streams, one named "Time Stamps" of type "gesture_info" with 3 channels (gesture ID, start time, end time), and the other named "Grip Power Time Stamp", used to mark the start and end of the persons max-grip calibration trial. Both publish dat at 500 Hz so it aligns with the EMG stream from the armband.

As the script runs, it cycles through a random list of gesture prompts: ```['open', 'power', 'pronate', 'supinate', 'rest', 'pinchgrasp']```, displaying each on-screen while simultaneously sending its corresponding ID and timing markers into the LSL network. 

```CollectEMGData.py`` subscribes to the same stream names as LSL inlets. All recorded data (EMG and gesture events) are synchronized when written to the `.xdf` file.

### 3. Recording Synchronization

Once the EMG and cue generation scripts are running, their data gets captured together in perfect temporal alignment, mostly handeled by **```train_and_run.py```*** using the **Lab Streaming Layer (LSL)** framework and the **LabRecorder** tool that ```train_and_run.py``` starts automatically.

when ```train_and_run.py``` starts, it launches both the cue display script (```ShowCues.py```) and the EMG acquisition script (```CollectEMGData.py```) in parallel threads:
```python
t1 = threading.Thread(target=start_showcues_script)
t2 = threading.Thread(target=start_collect_emg_data_script)
t1.start()
t2.start()
```

These 2 scripts run simultaneously and each registers its own **LSL outlet stream**, a small background server that broadcasts data on the local network. ```ShowCues.py``` broadcasts gesture marker data through “Time Stamps” and “Grip Power Time Stamp” outlets, while ```CollectEMGData.py``` publishes raw EMG samples through the “OyMotion GForce+” outlet. 

Every sample pushed into outletss through calls like: ```outlet.push_sample([...])``` is tagged with an accurate timestamp from ```pylsl.local_clock()```. 

All outlets use the same internal LSL clock, so their sample times are synchronized automatically even though the scripts run independently.

After those outlets are active, ```train_and_run.py``` launches **LabRecorder** by calling:
```python
lr = LabRecorderCLI(LABRECORDER_CLI_PATH)
lr.start_recording(filename, streamargs)
```
These lines use the *liesl* wrapper to start the LabRecorder executable as a seperate background process. LabRecorder begins searching for any active LSL outlets that match the names and types listed in ```streamargs```. Since LSL streams advertise themselves automatically over multicast, LabRecorder can discover and subscribe to them automatically.

Once connected, LabRecorder continouusly receives samples from all 3 outlets, automatically aligns them using their embedded timestamps and writes them into a single .xd file under `train/collected_data/` which is set in the arguments: `lr.start_recording(filename, streamargs)`.

When `ShowCues.py` finishes running, it drops a flag file `recording_done.flag` which signals `train_and_run.py` to call `lr.stop_recording()`. This stops the LabRecorder subprocesses and finalizes the synchroinized recording.

Every EMG sample, cue marker, and grip strength timestamp are now saved together in the `.xdf` file down to millisecond alignment. 

### 4. Data Preprocessing

All data now exists in the `.xdf` file that contains the 3 LSL streams: EMG, gesture cue timestamps, and grip-strength timestamps. This file perserves temporal alignmnet but must be converted into a **MATLAB** `.mat` dataset for further trainng and analysis.

The conversion process is handled by `train/ConvertEMGDataFormat.py`. When executed, it loads the `.xdf` file using the *pyxdf* library. This library can read the LabRecorder file format and return a list of stream dictionaries. 

Each dictionary contains metadata (stream name, type, sampling rate, and channel count) and a data array containing all samples and timestamps for that stream.

The script identifies the 3 steams of interest by name: 
- “OyMotion GForce+”: the 8-channel EMG data
- “Time Stamps”: the gesture cue markers
- “Grip Power Time Stamp”: the max-grip calibration markers

The EMG stream provides a continuous array of shape (N, 8), where each column represents one electrode channel sampled at 500 Hz. The cue marker stream provides event boundaries. Each of its samples is a 3 element vector: `[gesture_id, start_time, end_time]`.

The script iterates through the liset of gestures in the cue stream. For each gesture, it finds all the EMG samples whose timestamps fall between the gesture's `start_time` and `end_time`. That subset of the EMG array forms a segment corresponding to one complete trial of a gesture.

Using the same queue list defined in `ShowCues.py`, it assigns each extracted segment to the appropriate label.

Each segment is then stored in a Python dictionary keyed by gesture name. A seperate segmant is extracted from the "Grip Power Time Stamp" stream to represent the participant's maximum-grip trial. This segment is used later as a reference for power normalization, allowing the system to express grip intensity as a percentage of the persons maximum effort.

Once all segments have been extracted, the script packages them into a final data structure that is Python and MATLAB friendly. It uses `scipy.io.savemat()` to save the dataset in a standardized format called `EMG_data_YYY-MM-DD.mat`. 

It's written to both `train/collected_data/` and `control/collected_data/`
so the same dataset can be accessed by both the offline and runtime components of the system.

Inside the `.mat` file, the top-level variable is named *train*. This variable is a two-element cell array. 

1. Element 1: A structure containing all gesture-specific EMG recordings
   - Each key corresponds to a gesture name.
   - Each value is a 2D numeric array of shape (M, 8)
   - Each row contains one 8-channel EMG sample at 500 Hz rate
2. Element 2: A 2D array of shape (K, 8) representing the max grip segment used for normalization

In MATLAB, this appears as:
```matlab
train = 
{
  [1,1] = struct with fields:
              open:        [A1×8 double]
              power:       [A2×8 double]
              rest:        [A3×8 double]
              pinchgrasp:  [A4×8 double]
  [2,1] = [B×8 double]   % max grip segment
}
```
Inspecting it in Python, it appears as:
```python
{
  'train': [
      {'open': array(...), 
       'power': array(...),
       'rest': array(...),
       'pinchgrasp': array(...)},
      array(...)  # max grip segment
  ]
}
```

Accessing data for a specific gesutre example: 
```matlab
data = train{1}.power;  % returns the EMG matrix for the "power" gesture
```
```python
data = mat['train'][0][0]['power']
```

We now have an organized `.mat` file where each gesture's EMG signal exists as an independent array.

### 5. Feature Extraction

The raw EMG data needs to be transformed into compact numerical representations that capture the most important features for muscle activity patterns. 

Feature extraction is performed on windowed EMG segments and computes descriptive metrics from each channel.

Feature extraction logic is in:
- `train/TimeDomainFilter.py` (for offline training)
- `control/TimeDomainFilter.py` (for real-time prediction)

And is used directly inside of:
- `train/trainClassifierAndPredict.py`
- `control/trainClassifierAndPredict.py`

Each EMG window is passed into an instance of the `TimeDomainFilter` class, defined in `train/TimeDomainFilter.py`. This object computes the 5 time-domain features for all 8 channels at once.
```python
feat = td5.filter(win)
```
`win` is the current 125x8 EMG segment, and the `filter()` method returns a single flattened vector containing all 40 values (5 features * 8 channels).

Windowing occurs automatically before this step in the training and control scripts. Each window spans 125 samples (250 ms), advancing every 25 samples (50 ms) producing a continuous stream of overlapping segments to ensure smooth slices of EMG activity.

This happens in `control/trainClassifierAndPredict`:
```python
while idx + emg_window_size < num_samples:
    window = raw_data[idx:(idx+emg_window_size), :]
    time_domain = td5.filter(window).flatten()
    class_data.append(np.hstack(time_domain))
    idx += emg_window_step
```

The system extracts 5 classical time-domain features from each EMG channel to provide a compact summary of muscle activation strength, variability, and frequency characteristics.
```python
mav = np.mean(np.abs(x), axis=0)                      # Mean Absolute Value
var = np.var(x, axis=0)                               # Variance
wl  = np.sum(np.abs(np.diff(x, axis=0)), axis=0)      # Waveform Length
zc  = np.sum(np.sum(np.dstack([
         np.abs(x[1:,:]) > self.__eps_zc,
         np.abs(x[:-1,:]) > self.__eps_zc,
         np.multiply(x[1:,:], x[:-1,:]) < 0]), axis=2) == 3, axis=0)  # Zero Crossings
ssc = np.sum(np.sum(np.dstack([
         np.abs(np.gradient(x, axis=0)[1:,:]) > self.__eps_ssc,
         np.abs(np.gradient(x, axis=0)[:-1,:]) > self.__eps_ssc,
         np.multiply(np.gradient(x, axis=0)[1:,:],
                     np.gradient(x, axis=0)[:-1,:]) < 0]), axis=2) == 3, axis=0)  # Slope Sign Changes
```

- MAV (Mean Absolute Value): Overall muscle activation level.
- Variance: Signal power and spread.
- Waveform Length: Cumulative signal change, correlates with muscle contraction intensity.
- Zero Crossings (ZC): Frequency content estimate—counts how often the signal crosses zero.
- Slope Sign Changes (SSC): Detects rapid shape changes, capturing subtle muscle activity differences.

Each feature is calculated for all 8 channels and concatenated, giving 40 features per window (5 features * 8 channels).

This fixed length vector forms the input to the LDA classifier.

### 6. Model Training and Evaluation

After feature extraction, each EMG window is represented as a 40-dimensional feature vector that describes the muscle activity across all 8 channels. A **Linear Discriminant Analysys (LDA) model is used to map these feature vectors to the correct gesture class.

- `train/trainClassifierAndPredict.py` (for offline model training and testing, reads from the `.mat` created earlier)
- `control/trainClassifierAndPredict.py` (for real-time inference using the trained model, can train a fresh model or reuse an existing one in memory)

The `.mat` file is first loaded using `scipy.io.loadmat()`. Each gestures array is windowed, features are extracted, and then flattened into rows of the training matrix X. Each row corresponds to one 250 ms EMG segment, and a matching label in y identifies which gesture that segment came from.
- X shape: (num_windows, 40) - each row is one feature vector
- y shape: (num_windows, ) - integer gesture labels

The system trains an LDA classifier using scikit-learn:
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
mdl = LDA(
    n_components=min(n_classes - 1, n_features),
    priors=np.ones(n_classes) / n_classes,
    shrinkage='auto',
    solver='eigen'
)
mdl.fit(X_train, y_train)
```

This model projects the 40-dimensional feature space into a smaller, discrimant space where gestures are maximally seperated.

For evaluation the dataset is split into 70% training and 30% testing with stratification to perserve class balance.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)
```

After, the model's predictions are compared with the ground-truth labels:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y
)
```

Additionally each prediction passes through the `UniformVoteFilter` from, which smooths rapid flickering by requiring consecutive identical predictions before confirming a gesture/

Once trained, the model object (`mdl`) is kept in memory and passed into the runtm control loop. 

### 7. Real-Time Classification and Control

Streaming data from the armband, classifiying gestures in real time, and transimitting corresponding commands to the hand. This operates inside of the `control/` directory.

- `control/EMG_Control.py` real-time EMG streaming, feature extraction, and classificaiton.
- `control/miaComBle.py` listens for prediction over TCP and forwards them to the MIA hand over BLE.
- `control/gestureDict.py` and `control/command_filter.py` maps class indices to specific ASCII commands that the hand understands.

At startup, the most recent `.mat` file is loaded, and the LDA classifier is retrained in memory:
```python
mdl, td5, uni, CLASSES, max_grip = train_classifier(mac, emg_window_size, emg_window_step, mat_file, test=False)
```
This initializes everything needed for inference including:
-  mdl = LDA classifier
-  td5 = feature extractor (TimeDomainFilter)
-  uni = smoothing filter (UniformVoteFilter)
-  max_grip = calibration reference

The script connects to the OyMotion GForce+ armband:
```python
gforce, outlet, q = await connect_armband_and_start_streaming(ARMBAND_MAC_ADDRESS, usingLSL=False)
```
EMG samples stream at 500 Hz and are collected into a sliding buffer

Every 50 ms, a new 250 ms EMG window is proccessed:
```python
feat = td5.filter(win)
pred = int(mdl.predict(feat.reshape(1, -1))[0])
out  = uni.filter(pred)
```
The classifier produces a predicted gesture index and the vote filter ensures only stable results are acted upon.

`EMG_Control.py` opens a TCP connection to a local socket server on port 12345:
```python
server_address = ('localhost', 12345)
sock.connect(server_address)
sock.sendall(f"{out}\n".encode("utf-8"))
```

Each confirmed gesture index (out) is sent over this socket as a line of text.

`miaComBLE.py` runs as the socket listener. It accepts the incoming connections, reads those indeces, converts them into control stirngs, and transmitting them to the hand over BLE.

The TCP bridge keeps the ML loop which needs high throughput and low latency seperate from BLE operations (which are slower and prone to blocking).

Inside `miaComBle.py`, the receieved gesture ID is mapped and sent to the hand:
```python
gesture_id = int(line)
command = gestureDict.gesture_check(gesture_id)
await client.write_gatt_char(RX_ID, command.encode("utf-8"), response=True)
```
The hand interprets this commands and performs its corresponding motion.

### 8. Grip-Strength Scaling

Preliminary mechanism for grip-strength estimation, allowing the "Power" gesture to reflect muscle intensity instead of an on/off

The code is in `train/trainClassifierAndPredict.py` where the code computes a relativ power level by comparing the mean EMG amplitude of the current window to the user's recorded maximum grip segment (max_grip) from calibration:
```python
if gesture == 'POWER':
    adjusted_win = np.mean(np.abs(win))
    power_level = int(adjusted_win / max_grip * 100)
    print(str(f"{power_level:03}"))
```
I think it still needs to be implemented into the BLE control loop










