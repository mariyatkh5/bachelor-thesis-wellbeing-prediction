import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const


def generate_labels():
    # Load the NASA TLX data file
    file_path = os.path.join(const.BASE_DIR, 'dataset1_SenseCobot/Additional_Information/NASA_TLX.csv')
    nasa_tlx_data = pd.read_csv(file_path)
    nasa_tlx_data.columns = nasa_tlx_data.columns.str.strip()
    nasa_tlx_data = nasa_tlx_data.dropna(how='all')

    tasks = range(1, 6)  # Tasks 1–5
    my_data = []
    paper_data = []

    for participant_id, row in enumerate(nasa_tlx_data.iterrows(), start=1):
        row = row[1]
        for task in tasks:
            a_score = row[f'A_TASK {task}']
            b_score = row[f'B_TASK {task}']

            # --- Altes Schema (Summe, 2 Labels) ---
            total_score = a_score + b_score
            if 0 <= total_score < 6:
                my_label = 1
            elif 6 <= total_score < 14:
                my_label = 0
            else:
                my_label = None

            if my_label is not None:
                my_data.append({
                    'Participant': participant_id,
                    'Task': task,
                    'A_Score': int(a_score),
                    'B_Score': int(b_score),
                    'Total_Score': int(total_score),
                    'Well-being': my_label
                })

            # --- Paper-Schema (Durchschnitt, 4 Labels) ---
            mean_score = (a_score + b_score) / 2
            if 0 <= mean_score <= 1:
                paper_label = 0
            elif 2 <= mean_score <= 3:
                paper_label = 1
            elif 4 <= mean_score <= 5:
                paper_label = 2
            elif 6 <= mean_score <= 7:
                paper_label = 3
            else:
                paper_label = None

            if paper_label is not None:
                paper_data.append({
                    'Participant': participant_id,
                    'Task': task,
                    'A_Score': int(a_score),
                    'B_Score': int(b_score),
                    'Mean_Score': mean_score,
                    'Stress_Label': paper_label
                })

    # --- Save both ---
    output_folder = os.path.join(const.OUTPUT_DIR, "dataset1")
    os.makedirs(output_folder, exist_ok=True)

    my_df = pd.DataFrame(my_data)
    paper_df = pd.DataFrame(paper_data)

    my_file = os.path.join(output_folder, "labels.csv")
    paper_file = os.path.join(output_folder, "labels_paper.csv")

    my_df.to_csv(my_file, index=False)
    paper_df.to_csv(paper_file, index=False)

    print(f" Saved: {my_file} ({len(my_df)} rows)")
    print(my_df['Well-being'].value_counts())
    print(f" Saved: {paper_file} ({len(paper_df)} rows)")
    print(paper_df['Stress_Label'].value_counts())


if __name__ == "__main__":
    generate_labels()
