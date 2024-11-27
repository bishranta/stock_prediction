import requests
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime


def scrape_and_write_to_csv():
    url = 'https://www.sharesansar.com/live-trading'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')

    if table:
        rows = table.find_all('tr')[1:]  # Skip the header row
        current_date = datetime.now().strftime('%Y-%m-%d')

        for row in rows:
            cells = row.find_all('td')
            if len(cells) > 6:  # Ensure there are enough cells for each field
                # Extract relevant data matching the CSV structure
                stock_symbol = cells[1].text.strip()
                data = [
                    stock_symbol,  # Symbol
                    current_date,  # Date
                    cells[5].text.strip(),  # Open
                    cells[6].text.strip(),  # High
                    cells[7].text.strip(),  # Low
                    cells[2].text.strip(),  # Close
                    cells[4].text.strip(),  # Percent Change
                    cells[8].text.strip()  # Volume
                ]

                csv_filename = f"{stock_symbol}.csv"
                file_mode = 'a' if os.path.exists(csv_filename) else 'w'

                with open(csv_filename, mode=file_mode, newline='') as file:
                    writer = csv.writer(file)
                    if file_mode == 'w':
                        # Write header if file is new
                        writer.writerow(["Symbol", "Date", "Open", "High", "Low", "Close", "Percent Change", "Volume"])
                        print(f"CSV file '{csv_filename}' created with header.")
                    writer.writerow(data)
                    print(f"Data appended to '{csv_filename}'.")


scrape_and_write_to_csv()

# schedule.every().sunday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().monday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().tuesday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().wednesday.at("15:30").do(scrape_and_write_to_csv)
# schedule.every().thursday.at("15:30").do(scrape_and_write_to_csv)
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)