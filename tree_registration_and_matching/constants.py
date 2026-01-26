from pathlib import Path

import pyproj

LAT_LON_CRS = pyproj.CRS.from_epsg(4326)

DATA_DIR = Path(Path(__file__).parent, "..", "data").resolve()
