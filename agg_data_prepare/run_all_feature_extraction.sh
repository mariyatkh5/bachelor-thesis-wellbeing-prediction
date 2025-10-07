# NOTE make sure you source correctly
source ~/Users/mariya_ty/Desktop/BA projekt

echo $PWD
# dataset1
python chronopilot_feature_extraction/dataset1/ecg_features.py &
python chronopilot_feature_extraction/dataset1/eda_features.py &
python chronopilot_feature_extraction/dataset1/eeg_features.py &
python chronopilot_feature_extraction/dataset1/gsr_features.py &
python chronopilot_feature_extraction/dataset1/labels.py &
python chronopilot_feature_extraction/dataset1/tmp_features.py &

# dataset2
python chronopilot_feature_extraction/dataset2/eda_ecg_ppg_features.py &
python chronopilot_feature_extraction/dataset2/labels.py &


# dataset3
python chronopilot_feature_extraction/dataset3/ecg_features.py &
python chronopilot_feature_extraction/dataset3/gsr_features.py &
python chronopilot_feature_extraction/dataset3/labels.py &
python chronopilot_feature_extraction/dataset3/ppg_features.py &

# dataset4
python chronopilot_feature_extraction/dataset4/ecg_features.py &

# dataset5
python chronopilot_feature_extraction/dataset5/ecg_features.py &
python chronopilot_feature_extraction/dataset5/eda_features.py &
python chronopilot_feature_extraction/dataset5/labels.py &
python chronopilot_feature_extraction/dataset5/ppg_features.py &

# dataset8
python chronopilot_feature_extraction/dataset8/ecg_features.py &
python chronopilot_feature_extraction/dataset8/eda_features.py &
python chronopilot_feature_extraction/dataset8/labels.py &

# dataset9
python chronopilot_feature_extraction/dataset9/dataset9.py &

# dataset7
python chronopilot_feature_extraction/dataset7/filtered_data.py  # NOTE this needs to be completed first before we can continue
python chronopilot_feature_extraction/dataset7/gsr_features.py &
python chronopilot_feature_extraction/dataset7/labels.py &
python chronopilot_feature_extraction/dataset7/ppg_features.py &

wait