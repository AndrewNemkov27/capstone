import os
import csv
from collections import Counter

# locate the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, "realData.csv")


def update_word_frequencies(csv_file):
    # read CSV and count word occurrences
    word_counts = Counter()
    total_words = 0
    updated_rows = []

    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

        for row in rows:
            word = row[0]  # word_JAP column
            word_counts[word] += 1
            total_words += 1

    # Update word_freq column
    for row in rows:
        word = row[0]  # word_JAP column
        row[8] = round(word_counts[word] / total_words, 4)  # update word_freq column
        updated_rows.append(row)
        print(f"Success: Updated word '{word}' with frequency {row[8]}")

    # rewrite CSV with updated word frequencies
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(updated_rows)

    print("CSV file successfully updated with word frequencies!")

# run function
update_word_frequencies(csv_filename)
