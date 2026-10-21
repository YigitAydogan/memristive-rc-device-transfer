# Measured device data

The TiOx calibration and the figures based on it need the measured data of two published datasets. They are not
redistributed in this repository.

1. **TiOx pulse measurements.** Wang et al., *Temporal dynamics of TiOx memristor for reservoir computing applications*,
   J. Phys. D: Appl. Phys. 59 015101 (2026). Dataset: https://doi.org/10.5258/SOTON/D3702, licence CC BY.
   Download `Dataset_Figures.zip` and place either the zip or the CSV files it contains, `Dataset_Fig2.csv` to
   `Dataset_Fig6.csv`, in `data/measured/`.
2. **Mesoporous silica memristor measurements.** Jaafar et al., *3D-structured mesoporous silica memristors for
   neuromorphic switching and reservoir computing*, Nanoscale 14 17170 (2022). Dataset:
   https://doi.org/10.5258/SOTON/D2447. Place `Dataset_2.xlsx` in `data/measured/`.

The simulation scripts look for these files in `<project root>/data/measured/`, in `<project root>/data/` and in
`/content`, and the environment variable `MEMRISTIVE_MEASURED_DIR` can point to another folder. The figure scripts in
`analysis/` look in this folder.

The UCR time-series classification datasets are downloaded automatically through `aeon` on first use.
