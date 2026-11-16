"""Download and unpack MovieLens 1M into streaming_algorithms/data/ml-1m/ (OS independent)."""
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA = os.path.join(HERE, "data")
ZIP = os.path.join(DATA, "ml-1m.zip")

os.makedirs(DATA, exist_ok=True)
if os.path.exists(os.path.join(DATA, "ml-1m", "ratings.dat")):
    print("already downloaded")
else:
    print("downloading", URL)
    urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():  # extract only expected files (no path tricks)
            if name in ("ml-1m/ratings.dat", "ml-1m/movies.dat", "ml-1m/users.dat", "ml-1m/README"):
                z.extract(name, DATA)
    print("ok ->", os.path.join(DATA, "ml-1m"))
