import uuid

def get_machine_id():
    return hex(uuid.getnode())

if get_machine_id() == '0xf2925d19892a':  # tills laptop
    OUTPUT_DIR = "/Volumes/Data/chronopilot/external_datasets/agg_data"  # Path to base folder for the output files
    BASE_DIR = "/Volumes/Data/chronopilot/external_datasets"  # Path to the base folder for the input files
    FILTERED_DIR = "/Volumes/Data/chronopilot/external_datasets/filtered_data"  # Path to the base folder for the filtered files
    # Window settings in samples (72 sec window, 20 sec step)
    INTERVAL = 72
    STEP = 20
else:
    OUTPUT_DIR = "agg_data"  # TODO check if this is correct
    BASE_DIR = "/Users/mariya_ty/Desktop/BA/BA projekt"
  # Path to the base folder for the input files
    FILTERED_DIR = "filtered_data"  # TODO check if this is correct
    INTERVAL = 72
    STEP = 20
    
