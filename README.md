# tree-registration-and-matching

## Install

```
conda create -n tree-registration-and-matching python=3.12 -y
conda activate tree-registration-and-matching
```
Install poetry
```
poetry install
```

## Example data
An example benchmark is provided to test registration algorithms and can be downloaded from Box at this [link](https://ucdavis.box.com/v/ofo-tree-registration). This dataset consists of data from 232 drone imagery collections and corresponding field reference information. The field reference information was manually registered to the CHM product for each dataset. The data should be downloaded and placed inside of the `data` folder in this repository. You can download a subset of the CHM products to save space and time if desired.
* **CHMs:** The drone images were registered together using photogrammetry (Agisoft Metashape). This produces a digital surface model (DSM; top of canopy) and digital terrain model (DTM; bare earth model) for each site. Taking the differece of the two we obtain a raster representing the estimated canopy height model (CHM). This is cropped to a buffer of xx meters around the region which was surveyed. The data is provided as a geotiff file (`.tif`) which encodes the spatial location of the data.
* **field_trees.gpkg:** The locations of trees observed from ground surveys. This is provided as a geospatial vector file where each point represents a single tree. Many attributes from the field survey are provided but are not used in most registration approaches. The `dataset_id` field denotes which CHM the tree should be paired with and the `height` field represents the field-measured height (m). The data has undergone some pre-processing to reduce it to only trees that are expected to be visible from above. First, all trees that are dead (`live_dead == 'D'`) are removed since dead trees are reconstructed poorly by photogrammetry. Then trees are removed if they are likely to be under another larger tree. Starting with the tallest tree, any shorter trees are removed if they are within `1 + 0.1 * height (m)` of the tall tree. This process is repeated for all trees from tallest to shortest.
* **plot_bounds.gpkg:** The region surveyed in field represented as polygon vector data. The `dataset_id` column denotes which CHM and field trees the data should be paired with.
* **shift_quality.json:** This is a mapping from `dataset_id` to a number from 1-4. The latter is a quality score, with 4 being the highest. This takes into account how accurate the field survey appeared to be when compared to the CHM data. Furthermore, it also represents how confident a human annotator was in finding the correct shift for that dataset.
* **shfts_per_dataset.json:** All of the field trees and plot bounds have been shifted so that they align as well as possible with the CHM, as determined by a human annotator. This shift, represented as an (x, y) shift in meters in the `EPSG:xxxxx` coordinate frame, represents how much the data needed to be shifted by. Since the provided trees and plot bounds are already shifted, you must apply the negative of this value to get the initial location of the trees and plots.