import glob
from collections import defaultdict

root_dir = "../data/adl-piano-midi"

genres = defaultdict(int)

for f in glob.glob(root_dir + "/*/*/*/*", recursive=True):
    split = f.split("/")
    genres[split[3]] += 1

for key in sorted(genres, key=genres.get, reverse=True):
    print(key, genres[key])

print(sum(genres.values()))

