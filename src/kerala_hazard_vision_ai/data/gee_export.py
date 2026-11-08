"""Build before/after/conditioning stacks for each event in Google Earth Engine.

Earth Engine (``ee``) is imported inside the functions that need it, so the rest of the
package and the test suite work without ``earthengine-api`` installed. Call
``ee.Initialize(project=...)`` before using any function except the date-window helpers.

Datasets:
    * Sentinel-2 surface reflectance: ``COPERNICUS/S2_SR_HARMONIZED``
    * Elevation (for slope): ``COPERNICUS/DEM/GLO30``
    * Daily rainfall: ``UCSB-CHG/CHIRPS/DAILY``
    * Soil texture class: ``OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02``
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kerala_hazard_vision_ai.config import DataConfig, Event
from kerala_hazard_vision_ai.data.severity import BANDS

if TYPE_CHECKING:
    import ee

S2_COLLECTIONS = {
    "l2a": "COPERNICUS/S2_SR_HARMONIZED",  # surface reflectance; global only from 2019
    "l1c": "COPERNICUS/S2_HARMONIZED",  # top-of-atmosphere; global since 2015
}
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
DEM_COLLECTION = "COPERNICUS/DEM/GLO30"
RAIN_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
SOIL_IMAGE = "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02"

S2_REFLECTANCE_SCALE = 10_000  # Sentinel-2 digital numbers per unit reflectance (L1C and L2A)
# Scene-classification classes to mask: cloud shadow, medium/high cloud, thin cirrus.
S2_MASKED_SCL = (3, 8, 9, 10)
NODATA = -9999.0
FILE_DIMENSIONS = 4096  # one GeoTIFF per event (a 0.2° box is ~2200 px at 10 m)

MAX_PIXELS = 10**10  # Earth Engine request limit; far above one ~2200 px event box

DateWindow = tuple[date, date]


def _season(start_year: int, cfg: DataConfig) -> DateWindow:
    end_year = start_year + 1  # the season crosses New Year (checked by DataConfig)
    last_day = calendar.monthrange(end_year, cfg.dry_season_end_month)[1]
    return (
        date(start_year, cfg.dry_season_start_month, 1),
        date(end_year, cfg.dry_season_end_month, last_day),
    )


def dry_season_before(day: date, cfg: DataConfig) -> DateWindow:
    """The latest complete dry season (default 1 Dec - 30 Apr) that ends before ``day``."""
    end_year = day.year if day.month > cfg.dry_season_end_month else day.year - 1
    return _season(end_year - 1, cfg)


def dry_season_after(day: date, cfg: DataConfig) -> DateWindow:
    """The first dry season that starts after ``day``."""
    start_year = day.year if day.month < cfg.dry_season_start_month else day.year + 1
    return _season(start_year, cfg)


def composite_windows(event: Event, cfg: DataConfig) -> tuple[DateWindow, DateWindow]:
    """Inclusive date windows for the before and after composites.

    The before image is always the dry season before the event (no flood water, little
    cloud). Landslide scars stay visible for months, so their after image is the next dry
    season, which keeps the pair season-matched. Flood water drains within days or weeks, so
    flood events give their own near-event ``after`` window.
    """
    after = event.after or dry_season_after(event.date, cfg)
    return dry_season_before(event.date, cfg), after


def processing_level(hazard: str, cfg: DataConfig) -> str:
    """Sentinel-2 product for a hazard: ``"l1c"`` for those in ``cfg.toa_hazards``, else ``"l2a"``.

    Flood events need near-event imagery from 2018 onwards, where only Level-1C is global.
    Using one level per hazard keeps a hazard's training pairs, test pairs and demo inputs alike.
    """
    return "l1c" if hazard in cfg.toa_hazards else "l2a"


def aoi(bbox: tuple[float, float, float, float]) -> ee.Geometry:
    """Rectangle geometry from (min_lon, min_lat, max_lon, max_lat)."""
    import ee

    return ee.Geometry.Rectangle(list(bbox))


def _mask_clouds(image: ee.Image) -> ee.Image:
    import ee

    scl = image.select("SCL")
    clear = ee.Image.constant(1)
    for cls in S2_MASKED_SCL:
        clear = clear.And(scl.neq(cls))
    return image.updateMask(clear)


def _mask_clouds_cloud_score(clear_threshold: float):
    def mask(image: ee.Image) -> ee.Image:
        return image.updateMask(image.select("cs_cdf").gte(clear_threshold))

    return mask


def s2_composite(
    region: ee.Geometry,
    start: date,
    end: date,
    cfg: DataConfig,
    level: str = "l2a",
) -> ee.Image:
    """Cloud-masked median Sentinel-2 composite in reflectance units (0-1), ``BANDS`` only.

    Level-2A clouds are masked with its scene classification; Level-1C clouds with Google's
    Cloud Score+. A window without any image gives a fully masked composite (not an image
    without bands), so the cloud check scores it 0 instead of Earth Engine failing.
    """
    import ee

    collection = (
        ee.ImageCollection(S2_COLLECTIONS[level])
        .filterBounds(region)
        .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())  # end exclusive
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.max_cloud_pct))
    )
    if level == "l2a":
        collection = collection.map(_mask_clouds)
    else:
        collection = collection.linkCollection(
            ee.ImageCollection(CLOUD_SCORE_COLLECTION), ["cs_cdf"]
        ).map(_mask_clouds_cloud_score(cfg.cloud_score_clear))
    empty = ee.Image.constant([0] * len(BANDS)).rename(list(BANDS)).toUint16().updateMask(0)
    bands = collection.select(list(BANDS)).merge(ee.ImageCollection([empty]))
    return bands.median().divide(S2_REFLECTANCE_SCALE).clip(region)


def slope_image(region: ee.Geometry) -> ee.Image:
    """Terrain slope in degrees from the Copernicus 30 m elevation model."""
    import ee

    dem_collection = ee.ImageCollection(DEM_COLLECTION).filterBounds(region).select("DEM")
    dem = dem_collection.mosaic().setDefaultProjection(dem_collection.first().projection())
    return ee.Terrain.slope(dem).rename("slope")


def soil_image() -> ee.Image:
    """USDA soil texture class (1-12) of the topsoil."""
    import ee

    return ee.Image(SOIL_IMAGE).select("b0").rename("soil")


def conditioning(region: ee.Geometry, event: Event, cfg: DataConfig) -> ee.Image:
    """Three-band image: slope (degrees), rain (mm over the window), soil texture class."""
    import ee

    rain_start = event.date - timedelta(days=cfg.rain_days_before)
    rain = (
        ee.ImageCollection(RAIN_COLLECTION)
        .filterDate(rain_start.isoformat(), (event.date + timedelta(days=1)).isoformat())
        .select("precipitation")
        .sum()
        .rename("rain")
    )
    return ee.Image.cat([slope_image(region), rain, soil_image()]).toFloat().clip(region)


def event_stack(event: Event, cfg: DataConfig) -> ee.Image:
    """The 11-band stack expected by ``tiling.split_stack``: before, after, slope, rain, soil."""
    region = aoi(event.bbox)
    (pre_start, pre_end), (post_start, post_end) = composite_windows(event, cfg)
    level = processing_level(event.hazard, cfg)
    before = s2_composite(region, pre_start, pre_end, cfg, level)
    after = s2_composite(region, post_start, post_end, cfg, level)
    return (
        before.rename([f"pre_{b}" for b in BANDS])
        .addBands(after.rename([f"post_{b}" for b in BANDS]))
        .addBands(conditioning(region, event, cfg))
        .toFloat()
    )


def cloud_free_fraction(event: Event, cfg: DataConfig) -> dict[str, float]:
    """Share of the event area with a valid (cloud-free) composite pixel, per window."""
    import ee

    region = aoi(event.bbox)
    result = {}
    for name, (start, end) in zip(("pre", "post"), composite_windows(event, cfg), strict=True):
        composite = s2_composite(region, start, end, cfg, processing_level(event.hazard, cfg))
        valid = composite.select("B4").mask()
        stats = valid.reduceRegion(
            ee.Reducer.mean(), region, scale=cfg.check_scale_m, maxPixels=MAX_PIXELS
        )
        result[name] = float(stats.get("B4").getInfo() or 0.0)
    return result


def export_event(event: Event, cfg: DataConfig) -> ee.batch.Task:
    """Start a GeoTIFF export of the event stack to Google Drive and return the task."""
    import ee

    task = ee.batch.Export.image.toDrive(
        image=event_stack(event, cfg).unmask(NODATA),
        description=event.name,
        folder=cfg.drive_folder,
        fileNamePrefix=event.name,
        region=aoi(event.bbox),
        scale=cfg.scale_m,
        crs=cfg.crs,
        maxPixels=MAX_PIXELS,
        fileDimensions=FILE_DIMENSIONS,
        formatOptions={"noData": NODATA},
    )
    task.start()
    return task


def read_stack(path: Path) -> np.ndarray:
    """Read an exported GeoTIFF as float32 (bands, H, W) with no-data turned into NaN."""
    import rasterio

    with rasterio.open(path) as src:
        stack = src.read().astype(np.float32)
    stack[stack == NODATA] = np.nan
    return stack
