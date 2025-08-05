import os
import pickle
import pandas as pd
from imdb import IMDb
import time
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Config
TOP_N = 5320016  # Number of movies to scrape = 5320016
THREADS = 4
RETRY_LIMIT = 3
DELAY_RANGE = (0.8, 1.4)

TITLE_CACHE = "DATA/cache/titles.pkl"
MOVIE_CACHE = "DATA/cache/movie_cache.pkl"
PROGRESS_LOG = "DATA/cache/progress.log"
RAW_CSV = "DATA/raw/raw_data.csv"
TSV_PATH = "DATA/raw/title.basics.tsv"

# Ensure directories exist
os.makedirs("DATA/cache", exist_ok=True)
os.makedirs("DATA/raw", exist_ok=True)

# Check for TSV existence
if not os.path.exists(TSV_PATH):
    raise FileNotFoundError(f"Missing required TSV file: {TSV_PATH}")

# Load Titles
if os.path.exists(TITLE_CACHE):
    print("Loading cached titles...")
    with open(TITLE_CACHE, "rb") as f:
        top_titles = pickle.load(f)
else:
    print("Reading IMDb TSV file...")
    df = pd.read_csv(TSV_PATH, sep='\t', low_memory=False)
    movies = df[df['titleType'] == 'movie'][['tconst', 'primaryTitle']].dropna().reset_index(drop=True)
    top_titles = movies.head(TOP_N)
    with open(TITLE_CACHE, "wb") as f:
        pickle.dump(top_titles, f)
    print("Titles cached.")

# IMDbPY Setup
ia = IMDb()

# Load movie cache
if os.path.exists(MOVIE_CACHE):
    with open(MOVIE_CACHE, "rb") as f:
        movie_data = pickle.load(f)
else:
    movie_data = {}

# Resume Progress
start_index = 0
if os.path.exists(PROGRESS_LOG):
    with open(PROGRESS_LOG, "r") as f:
        try:
            start_index = int(f.read().strip())
        except ValueError:
            start_index = 0

# CSV Setup
csv_lock = Lock()
if not os.path.exists(RAW_CSV):
    with open(RAW_CSV, "w", newline='', encoding="utf-8") as raw_file:
        writer = csv.writer(raw_file)
        writer.writerow(["imdb_id", "title", "plot", "genres"])

# Locks for thread-safe operations
data_lock = Lock()
log_lock = Lock()

# Scraping Function
def scrape_movie(i):
    row = top_titles.iloc[i]
    imdb_id, title = row['tconst'], row['primaryTitle']

    with data_lock:
        if imdb_id in movie_data:
            return imdb_id, movie_data[imdb_id], i

    for attempt in range(RETRY_LIMIT):
        try:
            movie_id = imdb_id.replace("tt", "")
            movie = ia.get_movie(movie_id)
            plot = movie.get('plot', [None])[0]
            genres = ', '.join(movie.get('genres', []))
            data = (imdb_id, title, plot, genres)

            with data_lock:
                movie_data[imdb_id] = data

            time.sleep(random.uniform(*DELAY_RANGE))
            return imdb_id, data, i
        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                print(f"[{i}] Failed: {title} | Error: {e}")
            time.sleep(random.uniform(1.5, 2.0))
    return imdb_id, None, i

# Parallel Scraping
print(f"Starting scraping from index {start_index}...")

with ThreadPoolExecutor(max_workers=THREADS) as executor:
    futures = {executor.submit(scrape_movie, i): i for i in range(start_index, len(top_titles))}

    with open(RAW_CSV, "a", newline='', encoding="utf-8") as raw_file:
        writer = csv.writer(raw_file)

        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Scraping Movies"), start=start_index + 1):
            try:
                imdb_id, data, i = future.result()

                if data:
                    with csv_lock:
                        writer.writerow(data)

                # Save every 100 or at the end
                if count % 100 == 0 or count == len(futures):
                    with data_lock:
                        with open(MOVIE_CACHE, "wb") as f:
                            pickle.dump(movie_data, f)

                    with log_lock:
                        with open(PROGRESS_LOG, "w") as f:
                            f.write(str(i + 1))
            except Exception as e:
                print(f"Error for index {futures[future]}: {e}")

# Final Save
with data_lock:
    with open(MOVIE_CACHE, "wb") as f:
        pickle.dump(movie_data, f)

with log_lock:
    with open(PROGRESS_LOG, "w") as f:
        f.write(str(len(top_titles)))

print("All data scraped and saved.")

# Post-processing
print("Running post-processing...")
df_raw = pd.read_csv(RAW_CSV)
filtered = df_raw.dropna(subset=['plot'])

valid_entries = filtered[filtered['genres'].notna() & (filtered['genres'] != '')]
final_100k = valid_entries.tail(100000)
final_100k.to_csv("DATA/raw/test_data.csv", index=False)
print("test_data.csv created with last 100k valid entries")

plot_only = filtered[filtered['genres'].isna() | (filtered['genres'] == '')]
plot_only.to_csv("DATA/raw/test_data2.csv", index=False)
print("test_data2.csv created for entries with plot but no genre")
