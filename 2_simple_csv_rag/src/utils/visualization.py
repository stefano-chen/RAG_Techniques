import pandas as pd
from pathlib import Path

# visualize first n line of csv file
def show_csv(file_path: str | Path, n_rows: int = 5) -> None:
    csv_file = pd.read_csv(filepath_or_buffer=file_path)

    print(csv_file.head(n=n_rows))