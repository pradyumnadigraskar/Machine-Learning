# TMDB Top-Rated Movies Data Extraction

This script is used to extract top-rated movies' data from the TMDB (The Movie Database) API and save it into a CSV file for further analysis.

## Description

The script fetches data about the top-rated movies using the TMDB API. It iterates over multiple pages of results to collect data fields such as:
- **ID**: Unique identifier for the movie.
- **Title**: Title of the movie.
- **Overview**: A brief summary of the movie.
- **Release Date**: The release date of the movie.
- **Popularity**: A score indicating the movie's popularity.
- **Vote Average**: The average rating of the movie.
- **Vote Count**: The number of votes the movie has received.

The collected data is saved in a CSV file named `movies.csv`.

---

## Prerequisites

1. Python 3.x installed.
2. Required Python libraries:
   - `pandas`
   - `requests`
3. An active TMDB API key.

---

## Script Breakdown

### 1. Importing Required Libraries
```python
import pandas as pd
import requests
```
- `pandas`: Used for creating and manipulating data in a tabular format.
- `requests`: Used for making HTTP requests to the TMDB API.

### 2. Fetching Data from the TMDB API (Single Page Example)
```python
response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1')
temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
df.head()
```
- Fetches top-rated movies data for a single page.
- Converts the response JSON into a pandas DataFrame.
- Filters specific columns of interest.

### 3. Iterating Over Multiple Pages
```python
df = pd.DataFrame()
for i in range(1,429):
    response = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page={}'.format(i))
    temp_df = pd.DataFrame(response.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
    df = df.append(temp_df, ignore_index=True)
```
- Loops through all available pages (from 1 to 428).
- Collects the results from each page and appends them to a single DataFrame `df`.
- Uses the `append()` method with `ignore_index=True` to maintain proper indexing.

### 4. Saving the Data to a CSV File
```python
df.to_csv('movies.csv')
```
- Saves the entire DataFrame `df` into a CSV file named `movies.csv`.

### 5. Checking the DataFrame Shape
```python
df.shape
```
- Outputs the number of rows and columns in the final DataFrame.

---

## Output
- A CSV file named `movies.csv` containing the top-rated movies' data.
- The file can be used for further analysis or visualization.

---

## Notes
1. Replace the `api_key` parameter in the URL with your own TMDB API key.
2. The API key used in this example is for demonstration purposes only and might not work.
3. Make sure you adhere to TMDB’s API usage policies and rate limits.

