import csv
import os


def store_winrates_money_in_csv(file_path: str, epoch: int, winrates: str, money: str):
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", -1, "utf_8") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["Epoch", "Winrates", "Money"])

        writer.writerow([epoch, winrates, money])
