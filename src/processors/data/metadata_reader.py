import csv
import os.path

class MetadataReader(object):

    def __init__(self):
        basedir = os.path.split(__file__)[0]
        self.path = os.path.join(basedir,"globolakes-static_lake_centre_fv1.csv")
        self.metadata = {}

    def load(self):
        f = open(self.path,"r",encoding="iso-8859-1")
        r = csv.reader(f)
        row = 0
        for line in r:
            row += 1
            # first 42 rows are comments
            if row < 43:
                continue

            s = line[0].strip()
            if s == "end_data":
                break
            lake_id = int(s)
            name = line[1].strip()
            country = line[2].strip()
            lat = float(line[3].strip())
            lon = float(line[4].strip())
            self.metadata[lake_id] = { "name":name, "country":country, "lat":lat, "lon":lon }
        f.close()

    def getMetadata(self,lake_id):
        return self.metadata[int(lake_id)]