# Tunnel Void Finder Documentation

This repository contains a custom pipeline for identifying and characterizing **tunnel voids** in weak lensing (WL) maps. The algorithm is tailored to work with convergence maps, applying noise suppression, SNR enhancement, and morphology-based detection strategies. It is optimized for use with simulated WL maps containing Gaussian shape noise (GSN) and includes flexible control over thresholds and filtering parameters.

---

## Main Notebook: `Tunnel_void_finder.ipynb`

This notebook runs the full tunnel void detection pipeline:

- This pipeline is designed for convergence maps with or without Gaussian shape noise (GSN), without requiring 3D galaxy positions.
- Applies Gaussian smoothing to suppress pixel-level noise and enhance extended underdensities, using a user-defined kernel width (in arcmin)
- Converts the convergence map to a signal-to-noise ratio (SNR) field.
- Applies threshold-based segmentation to identify candidate underdense regions.
- Tracks local minima across multiple SNR levels to define void centers.
- Computes void radii by growing circles until the average SNR crosses zero.
- Cleans the resulting catalog to remove overlapping or nested detections.
- Outputs void catalogs and statistical observables like the PDF and angular power spectrum.
- Future extensions will include tomographic binning and redshift-dependent void statistics.
- The code is modular and allows custom SNR thresholds and connectivity settings.

---- .

## Core Functions Used in `Tunnel_void_finder.ipynb`

### `cleaning_ambient()`
This function cleans the working directory and resets all files associated with a specific void-finding run. It ensures that intermediate outputs from previous runs are deleted before initiating a new analysis, avoiding contamination from stale data.

### `connected_regions()`
Wrapper function that coordinates several steps to compute voids across a range of SNR thresholds: smoothing and adding noise to the convergence map, computing the SNR map, and applying criteria for detecting connected regions of pixels in SNR maps. It outputs the cleaned SNR map, the original and processed convergence maps, and metadata about the field of view and resolution.

### `map_PDF_and_Pl()`
Computes and plots two key statistical observables from the WL maps: the one-point probability distribution function (PDF) of pixel values, and the angular power spectrum (Pl) of the convergence field, both before and after the application of GSN and smoothing. It uses FFTs to quantify how the WL signal is affected by noise, and saves the binned statistics to text files for further analysis.

### `void_minima()`
Implements a hierarchical center-finding procedure to identify candidate tunnel voids by tracking the persistence of local minima across multiple SNR thresholds. Outputs include the coordinates of potential void centers and the filtered SNR map.

### `void_radii()`
This function determines the effective radius of each tunnel void by iteratively expanding circular regions around identified minima until a SNR threshold is reached. It then performs a cleaning step to remove overlapping voids and saves the final catalog. A summary plot is also generated, overlaying detected voids on the SNR map for visual inspection.

### `map_vsf()`
Calculates the void size function (VSF), i.e. the number density of the detected tunnel voids in the input map as a function of their effective radius. It uses logarithmic or linear binning and returns the bin edges, VSF values, and associated errors. Optionally, it generates a plot of the VSF distribution. 

### `create_stacks_folders()`
Divides the void sample of the map into equi-populated radius bins and creates separate folders to store stacked tangential shear profiles. It facilitates ensemble averaging over voids of similar size and organizes data for efficient batch processing.

### `shells_profile()`
Computes the tangential shear profiles for individual tunnel voids by analyzing concentric radial shells around their centers. It calculates both differential and cumulative convergence, applies error propagation, and stores the resulting profiles in text files. Optionally, it plots the reduced tangential shear profiles for visual inspection.

### `process_shear_map()`
Generates and visualizes the stacked tangential shear profiles of WL tunnel voids, grouped by size bins. It reads individual void shear data, averages the profiles, and computes error estimates using covariance, jackknife, and bootstrap methods. It then saves the results and optionally plots the mean and binned shear signals.

---

## Additional Core Functions in `Finder_functions.py`

### `noise_and_smooth()`
Adds Gaussian shape noise to the input convergence map and applies Gaussian smoothing using the specified kernel size (in pixels). Returns both the smoothed noisy map and the original smoothed map. This step enhances large-scale structures while reducing pixel-level fluctuations.

### `find_peaks()`
Identifies connected regions below a given threshold in a 2D field (usually an SNR map). Labels each region using either 4- or 8-connectivity and returns a matrix of labeled regions along with their pixel indices. With case=1 define peaks (overdense regions), while with case=-1 find valleys (underdense regions).

### `clean()`
Cleans the void catalog of connected regions by comparing centers of void candidates. If two voids are closer than the specified cutoff (in units of radius), the smaller one is discarded. This prevents double-counting and enforces void separation.

### `calculate_max_min_r()`
Extracts the approximate maximum and minimum void radii from the input array. The maximum radius is computed as the first element rounded up, and the minimum as the last element.

### `size_function()`
Computes the VSF by binning void radii between specified limits using either linear or logarithmic spacing. It estimates Poisson errors and optionally generates bar or point plots with error bars. The function returns bin centers, VSF counts, and their uncertainties.

### `bin_prof()`
Binning function for building tangential shear profiles. nbins set the number of bins, ngal number of background galaxies, for the error sigmagal is the intrinsic ellipticity distribution. When cum=1, it enables the computation of the cumulative convergence, in addition to the differential one, within a circular area defined by the radius of each considered shell.

### `calculate_covariance_matrix()`
Computes the covariance and correlation matrices from an ensemble of tangential shear profiles and optionally plots them.

### `calculate_stat_error()`
Calculates and compares statistical errors on the stacked tangential shear profiles using multiple techniques:
-Standard deviation from the covariance matrix
-Jackknife resampling
-Bootstrap resampling
-Hartlap-corrected inverse covariance matrix

---

## Output Files

- `voids_radii.txt`: Final catalog with center, radius, and minimum SNR
- `PDF.txt`, `Pl.txt`: WL map statistics (distribution and power spectrum)
- Intermediate region statistics per threshold

---

## Example Usage

Edit and run the notebook `Tunnel_void_finder.ipynb`:

folderout = "your/output/path/"
smooth_filter_set = 2.5  # arcmin
i_fil = "01"            # map identifier

---

## Notes

If you use this void finder or any of its components in your work, please cite:

**Maggiore et al. (2025)**,  
*Weak-lensing tunnel voids in simulated light-cones: a new pipeline to investigate modified gravity and massive neutrinos signatures*,  
arXiv:2504.02041  
[https://arxiv.org/abs/2504.02041](https://arxiv.org/abs/2504.02041)

### BibTeX:

@ARTICLE{2025arXiv250402041M,
       author = {{Maggiore}, Leonardo and {Contarini}, Sofia and {Giocoli}, Carlo and {Moscardini}, Lauro},
        title = "{Weak-lensing tunnel voids in simulated light-cones: a new pipeline to investigate modified gravity and massive neutrinos signatures}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = apr,
          eid = {arXiv:2504.02041},
        pages = {arXiv:2504.02041},
          doi = {10.48550/arXiv.2504.02041},
archivePrefix = {arXiv},
       eprint = {2504.02041},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250402041M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


For questions, please contact the repository author or open an issue on GitHub.