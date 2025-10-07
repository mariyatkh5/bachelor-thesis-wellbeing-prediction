# Aggregating a Well-Being Dataset

## Overview

This project aggregates physiological datasets and extracts features to develop a robust classifier for well-being states. Physiological signals such as ECG, EDA, PPG, EEG, EMG and skin temperature are processed and labeled based on established psychological models like PANAS, NASA-TLX, and the Russel Circumplex Model.

---

## Features

- **Signal Processing**: Supports ECG, EDA, PPG, EEG, and skin temperature.
- **Feature Extraction**: Time-domain, frequency-domain, and nonlinear metrics.
- **Well-Being Classification**: Labels generated from psychological models.
- **Integration**: Aggregates data from multiple datasets into a unified structure.

---

## Datasets Used

1. **SenseCobot**  
   - **Focus**: Stress and cognitive load during collaborative robotics tasks.  
   - **Signals**: ECG, GSR, body temperature, EDA, EEG.  
   - **Data source**: [Zenodo Link](https://zenodo.org/records/10124005)  
   - **Publication**: [DOI: 10.1145/3610977.3636440](https://doi.org/10.1145/3610977.3636440)



2. **MAUS**  
   - **Focus**: Mental load using N-back tasks.  
   - **Signals**: ECG, PPG, GSR.  
   - **Data source**: [IEEE Dataport](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor)  
   - **Publication**: [ResearchGate](https://www.researchgate.net/publication/355925184)

3. **Simultaneous Physiological Measurements with Five Devices**  
   - **Focus**: Signal comparison under different cognitive and physical loads.  
   - **Signals**: ECG (collected during rest, walking, and tasks).  
   - **Data source**: [PhysioNet](https://physionet.org/content/simultaneous-measurements/1.0.2/)  
   - **Publication**: [DOI: 10.1371/journal.pone.0274994](https://doi.org/10.1371/journal.pone.0274994)

4. **Detection of Acute Stress Based on Physiological Signals**  
   - **Focus**: Stress levels during cognitive tasks.  
   - **Signals**: ECG, PPG, EDA.  
   - **Data source**: [IEEE Dataport](https://ieee-dataport.org/open-access/database-cognitive-load-affect-and-stress-recognition)  
   - **Publication**: [DOI: 10.11591/eei.v10i5.3130](https://doi.org/10.11591/eei.v10i5.3130)

5. **Daily Ambulatory Psychological and Physiological Recording**  
   - **Focus**: Emotional states over five days.  
   - **Signals**: Heart rate, skin response.  
   - **Data source**: [Synapse](https://www.synapse.org/#!Synapse:syn22418021/wiki/605529)  

6. **POPANE**  
   - **Focus**: Emotional states and motivational tendencies.  
   - **Signals**: ECG, EDA, temperature, respiration, ICG.  
   - **Data source**: [OSF Link](https://osf.io/94bpx/)  
   - **Publication**: [Scientific Data](https://doi.org/10.1038/s41597-021-01117-0)

7. **Physiological Responses to Vocal Emotions**  
   - **Focus**: Laughter and crying stimuli.  
   - **Signals**: ECG, EMG, EDA.  
   - **Data source**: [OSF Link](https://osf.io/nvb2u/)  

---


## 🛠️ Requirements



### Install Dependencies

Run the following command to install required libraries:

```bash
pip install numpy pandas scipy matplotlib wfdb neurokit2
```

---

## 📂 Project Structure

```plaintext
project/
├── dataset7/
│   ├── filtered_data.py    # Prepares necessary folders and filtered data
│   ├── ecg_features.py     # Extracts ECG features
│   ├── gsr_features.py     # Extracts GSR features
│   ├── ppg_features.py     # Extracts PPG features
│   ├── labels.py           # Processes well-being labels
├── dataset1/
│   ├── ecg_features.py     # Additional ECG feature scripts
├── dataset9/
│   ├── ecg_all_clean.py    # Cleans and integrates all ECG data
├── agg_data/               # Output directory for aggregated data
└── README.md               # Project documentation
```

---

## Usage

### Step 1: Prepare Data
Navigate to `dataset7` and run `filtered_data.py` to create necessary folders and prepare filtered data:

```bash
cd dataset7
python filtered_data.py
```

### Step 2: Extract Features
Run the respective scripts to extract features from physiological signals.

#### Example: ECG Feature Extraction
```bash
python dataset1/ecg_features.py
python dataset7/ecg_features.py
```

#### Example: GSR Feature Extraction
```bash
python dataset7/gsr_features.py
```

#### Example: PPG Feature Extraction
```bash
python dataset7/ppg_features.py
```

### Step 3: Generate Labels
Generate well-being labels using `labels.py`:


### Step 4: Integrate and Clean ECG Data
Once all ECG feature files are processed, finalize by running `ecg_all_clean.py`:


---

## Output Directory Structure

All aggregated and processed data will be stored in `agg_data/`:
```plaintext

agg_data/
├── dataset1/
│   ├── ecg_features.csv
│   ├── labels.csv
├── dataset7/
│   ├── ecg_features.csv
│   ├── gsr_features.csv
│   ├── labels.csv
├── dataset9/

```
---


Feel free to provide feedback or open issues for enhancements. 
