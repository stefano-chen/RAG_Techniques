from utils.visualization import show_csv
from pathlib import Path

# constant definition
CSV_PATH = Path("./data/customers-100.csv")

# ENTRYPOINT
if __name__ == "__main__":
    show_csv(file_path=CSV_PATH, n_rows=5)