import numpy as np
import math
import random as rnd
import warnings
import imageio
from astropy.io import fits
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, MultipleLocator, FixedLocator, FixedFormatter, LogLocator, LogFormatter
import matplotlib.cm as cm
import skimage.filters
import skimage.color
import skimage.io
from skimage import data, filters, measure, morphology
from skimage import segmentation
from skimage.measure import find_contours, approximate_polygon,subdivide_polygon
import numpy.distutils.ccompiler
from astropy.convolution import convolve, convolve_fft
import scipy.fftpack as fftengine
from mpl_toolkits.axes_grid1 import make_axes_locatable
import camb
import sys, os, tempfile
import glob
import re
import magic
import shutil
from astropy.units import deg
import cv2
from getdist import plots, MCSamples, loadMCSamples, types
from getdist.types import ResultTable
from scipy.stats import norm
import IPython

from matplotlib.path import Path

def copy_files(source_folder, destination_folder, arg, n):
    """
    Copy files from source folder to destination folder.

    Args:
    - source_folder (str): Path of the source folder containing files.
    - destination_folder (str): Path of the destination folder to copy files.
    - number of subfolders (the same in the source and the destination path)

    Returns:
    - None
    """

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print("Source folder does not exist.")
        return
    
    # Ensure the destination folder exists, or create it if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    subfolderss = [str(i).zfill(2) for i in range(n)]

    for l_c in subfolderss:
        # Create complete path of source folder
        source_path = os.path.join(source_folder, l_c)
        
        # Create complete path of destination folder
        destination_path = os.path.join(destination_folder, l_c)
        
        # Check the existence of destination folder
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        # Find all the files in the source folder
        files = glob.glob(os.path.join(source_path, arg))
        #files = glob.glob(os.path.join(source_path, "*.fits"))
        
        # Copy files to the destination folder
        for file in files:
            shutil.copy(file, destination_path)
            
def eliminate_files(source_folder, arg, n):
    """
    Eliminate files in the source folder that match the specified pattern.

    Args:
    - source_folder (str): Path of the source folder containing files.
    - arg (str): Filename pattern to be removed from source folder.
    - n (int): Number of subfolders in the source and destination paths.

    Returns:
    - None
    """

    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print("Source folder does not exist.")
        return
    
    subfolderss = [str(i).zfill(2) for i in range(n)]

    for l_c in subfolderss:
        # Create complete path of source folder
        source_path = os.path.join(source_folder, l_c)
        
        # Find all the files in the source folder matching the pattern
        files = glob.glob(os.path.join(source_path, arg))
        
        # Delete files from the source folder
        for file in files:
            os.remove(file)

def create_ambient(pre_path, n, cosmo):
    folders_to_create = ["Code", "input_relative", "output_relative"]
    for folder_name in folders_to_create:
        folder_path = os.path.join(pre_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

    subfolders = [str(i).zfill(2) for i in range(n)]

    # Se cosmo è una stringa, esegue il codice normalmente
    if isinstance(cosmo, str):
        for folder_name in ["input_relative", "output_relative"]:
            folder_path = os.path.join(pre_path, folder_name)
            for subfolder_name in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)

    # Se cosmo è una lista, crea una cartella per ogni elemento di cosmo
    elif isinstance(cosmo, list):
        for folder_name in ["input_relative", "output_relative"]:
            folder_path = os.path.join(pre_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            for c in cosmo:
                cosmo_folder_path = os.path.join(folder_path, c)
                os.makedirs(cosmo_folder_path, exist_ok=True)

                # Crea le n sottocartelle dentro ciascuna cartella di cosmo
                for subfolder_name in subfolders:
                    subfolder_path = os.path.join(cosmo_folder_path, subfolder_name)
                    os.makedirs(subfolder_path, exist_ok=True)

def cleaning_ambient(folderout, search, i_fil):
    subfolders = [dirpath for dirpath, _, _ in os.walk(folderout)]
    keep_subfolders = [subfolder for subfolder in subfolders if any(name in subfolder for name in search)]

    for subfolder in subfolders:            
        output_files = [file for file in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, file))]
        
        for file in output_files:
            file_path = os.path.join(subfolder, file)
            file_extension = os.path.splitext(file_path)[1]
            file_type = magic.from_file(file_path, mime=True)
            if (file_type == 'text/plain' and file_extension != '.txt') or (file_extension == '.txt'):
                os.remove(file_path)

    for subfolder in subfolders:
        if subfolder != folderout:
            if subfolder not in keep_subfolders:
                shutil.rmtree(subfolder)
        
    for search_name in search:
        search_folder = f"{search_name}_{i_fil}k"
        search_path = os.path.join(folderout, search_folder)
        os.makedirs(search_path, exist_ok=True)

def create_equibins(data, min_r, max_r, num_folders):
    if num_folders < 2:
        raise ValueError("The number of folders must be at least 2")

    num_elements_per_folder = len(data) // num_folders

    folder_edges = []

    for i in range(1, num_folders):
        index = i * num_elements_per_folder
        folder_edges.append(round(data[index], 2))
        
    ### Print stacks ####    
    if any(edge in data for edge in folder_edges):
        for i in range(num_folders):
            start_index = i * num_elements_per_folder
            end_index = start_index + num_elements_per_folder + 1
            if i == num_folders - 1:
                end_index = len(data)
            if i == 0:
                print(f"Stack {i + 1}: {data[start_index:end_index]}, number of elements: {end_index - start_index}")
            else:
                print(f"Stack {i + 1}: {data[start_index+1:end_index]}, number of elements: {len(data[start_index+1:end_index])}")
            
    else:
        for i in range(num_folders):
            start_index = i * num_elements_per_folder
            end_index = start_index + num_elements_per_folder
            if i == num_folders - 1:
                end_index = len(data)
            print(f"Stack {i + 1}: {data[start_index:end_index]}, number of elements: {end_index - start_index}")        
  
    equibins = [min_r] + folder_edges + [max_r]
            
    return equibins

def create_stacks_folders(folderout, min_r, max_r, i_fil, smooth_filter_set, num_folders, manual=True):
    import Finder_functions as mystery 
    radii = np.genfromtxt(folderout + 'voids_radii.txt', usecols=(2), unpack=True, skip_header=1)
    radii = np.sort(radii)
    #print(radii)

    if manual:
        equibins = mystery.create_equibins(radii, min_r, max_r, num_folders)
        #print(equibins)
    else:
        equibins = [1, 3.79, 4.85, 15]

    folder_names = []

    for i in range(len(equibins) - 1):
        folder_names.append(f'voids_{i_fil}k_SNR{smooth_filter_set}_{equibins[i]}-{equibins[i+1]}')

    for dir_name in folder_names:
        dir_path = os.path.join(folderout, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
    return equibins, folder_names        
        
def write_map(fileout,data,fov,zs,npixel):
    """
    writing map routine
    """
    hdu = fits.PrimaryHDU(data=data)
    header = hdu.header
    header['PHYSICALSIZE'] = fov
    header['REDSHIFTSOURCE'] = zs
    header.tofile(fileout,overwrite=True)
    hdu.writeto(fileout,overwrite=True)

def read_map(filin):
    """
    read the map produced my MapSim and BornApp post processing pipeline
    """
    kappa = fits.getdata(filin, ext=0)
    hdul = fits.open(filin)
    #print(hdul[0].header)
    hdr = hdul[0].header
    fov = hdr['HIERARCH PHYSICALSIZE']
    print('fov [deg] = ', fov)
    npix = hdr['NAXIS1']
    print('number of pixels = ', npix)
    zs = hdr['HIERARCH REDSHIFTSOURCE']
    print('zs = ', zs)
    fov_arcmin = fov*60.0
    pixel_size_arcmin = fov_arcmin/npix
    print('pixel size [arcmin] = ', pixel_size_arcmin)
    return kappa, zs, npix, pixel_size_arcmin

def GetShear(image_file,zs,fileDl,fov_arcsec,omegaM,omegaL,h):
    # alpha = 0
    cmd = " source ~/.bashrc &&  echo " + str(omegaM) + " " + str(omegaL) + " " + str(h) + " " + str(zs) + " " + str(fov_arcsec) + " " + str(image_file) + "  " + str(fileinDl) + "  0 | ~/bin/LensMOKALike-beta"
    os.system(cmd)
    return shear_map

def smooth(map, sigma_in_pix):
    """
    smooth the map with a gaussian filter
    this match the Gaussian filter definition by Lin & Kilbinger 2014 & Van Waerbeke 2000
    W ~ exp(-\theta^2/\sigma^2) while python does G ~ exp(-0.5 \theta^2/\sigma^2)
    """
    #print('filter of the smooth =', sigma_in_pix/np.sqrt(2)) #smootha con un filtro che ha varianza sigma_arcmin=1, 2.5, 5 convertito in unità di pixel perchè lo applica sulla mappa, e diviso per sqrt(2) perchè in scipy.ndimage.gaussian_filter() compare 2 al denominatore che deve essere semplificato per ricondurci al filtro di Lin & Kilbinger 2014 eq. 16
    smoothed_map=scipy.ndimage.gaussian_filter(map, sigma_in_pix/np.sqrt(2))
    return smoothed_map

def noise_and_smooth(filin, sigma_ell=0.3, sigma_arcmin=1, ngal=30, seed=1234):
    """
    add gaussian shape noise and smooth the map according to a particular filter
    """
    kappa, zs, npix, pixel_size_arcmin = read_map(filin)
    fov_arcmin = npix*pixel_size_arcmin
    print("fov [arcmin]", fov_arcmin)
    
    sigma_pixel = sigma_arcmin*npix/fov_arcmin #converte in unità di pixel la larghezza della funzione filtro gaussiana che verrà usata per smoothare
    
    # eq. 18 Lin & Kilbinger 2014 see Van Waerbeke (2000) formula valida per descrivere il rumore finale se si assumono scorrelate tra loro le ellitticità delle galassie sorgenti
    sigma_noise = np.sqrt(sigma_ell**2/(4*np.pi*ngal*sigma_arcmin**2)) #calcola la std che descrive il Gaussian random field smoothato N(\theta)
    print('sigma_noise in arcmin (final N)=', sigma_noise)
        
    #crea mappa GSN:
    np.random.seed(seed)
    # GSN per pixel eq. 20 Lin & Kilbinger 2014 see Van Waerbeke (2000)
    sigma_noise_pix = np.sqrt(sigma_ell**2/(2*ngal*pixel_size_arcmin**2)) #calcola la std della distribuzione gaussiana che  descrive #n(\theta)
    print('sigma GSN per pixel =', sigma_noise_pix) 
    #print('sigma_pixel = ', sigma_pixel)
    gaussianNoise = np.random.normal(0.0, sigma_noise_pix, npix**2) #genera npix^2 (2048x2048) valori random (1 per ogni pixel) presi da una distribuzione gaussiana con media=0 e varianza=sigma_noise_pixel^2
    gaussianNoise = np.reshape(gaussianNoise, (npix, npix)) #mappa GSN n(\theta)
    
    kappa_noised = kappa + gaussianNoise #k_n(\theta) = aggiunge mappa GSN n(\theta) a mappa convergenza originale k(\theta)
    
    #Smoothing:
    kappa_noised_smoothed = smooth(kappa_noised, sigma_pixel) #K_N(\theta) = fa lo smoothing della mappa k_n(\theta) per ridurre l'impatto del GSN
    kappa_smoothed = smooth(kappa, sigma_pixel) #K(\theta) = mappa k(\theta) smoothata
    gaussianNoise_smoothed = smooth(gaussianNoise, sigma_pixel) #N(\theta) = mappa GSN n(\theta) smoothata
    print('sigma of the final smooth filter in pixel =', sigma_pixel/np.sqrt(2))
    print()
    # print('mean (kappa O)  = ', np.mean(kappa.flatten()))
    # print('std (kappa O)  = ', np.std(kappa.flatten()))
    # print('mean (kappa S)  = ', np.mean(kappa_smoothed.flatten()))
    # print('std (kappa S)  = ', np.std(kappa_smoothed.flatten()))
    # print('std (kappa N)  = ', np.std(kappa_noised.flatten()))
    #print('std (final kappa N and S) = ', np.std(kappa_noised_smoothed.flatten()))
    # print('mean (only N)', np.mean(gaussianNoise.flatten()))
    ##print('std (only N)', np.std(gaussianNoise.flatten())) #molto simile a sigma_noise_pix
    # print('mean (only N and S)', np.mean(gaussianNoise_smoothed.flatten()))
    ##print('std (only N smoothed)', np.std(gaussianNoise_smoothed.flatten())) #molto simile a sigma_noise
    return kappa_noised_smoothed, kappa, fov_arcmin, sigma_noise, npix, zs


def plot_three_maps(kappa, final_kappa, SNR, fov_arcmin):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Plot three maps:
    - kappa
    - kappa_noised_smoothed (final_kappa)
    - SNR
    """
    vmin = -0.0247215488  # Minimum value for the color scale
    vmax = 0.0247215488  # Maximum value for the color scale
    ranges = fov_arcmin / 2.0  # Field of view in arcminutes
    extent = [-ranges, ranges, -ranges, ranges]
    cmap = cm.coolwarm  # Color map
    scaled_SNR = np.interp(SNR, (vmin, vmax), (0, 1))
    summary_image = cmap(scaled_SNR)

    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    plt.figure(figsize=(60, 20))

    # Primo subplot: Original kappa
    ax1 = plt.subplot(1, 3, 1, aspect='equal')
    ax1.set_xlabel(r"$\mathrm{R.A. \; [arcmin]}$", fontsize=70)
    ax1.set_ylabel(r"$\mathrm{Dec. \; [arcmin]}$", fontsize=70)
    ax1.tick_params(axis='both', labelsize=60)
    ax1.set_title("Original", fontsize=70)
    im1 = ax1.imshow(kappa, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.2)  # Colorbar accanto
    cb1 = plt.colorbar(im1, cax=cax1)
    cb1.set_label(r"$\mathsf{\kappa}$", rotation=0, fontsize=70)#, labelpad=5)
    cb1.ax.tick_params(labelsize=60)

    # Secondo subplot: Noised and smoothed kappa
    ax2 = plt.subplot(1, 3, 2, aspect='equal')
    ax2.set_xlabel(r"$\mathrm{R.A. \; [arcmin]}$", fontsize=70)
    ax2.set_ylabel(r"$\mathrm{Dec. \; [arcmin]}$", fontsize=70)
    ax2.set_title("Noised and Smoothed", fontsize=70)
    ax2.tick_params(axis='both', labelsize=60)
    im2 = ax2.imshow(final_kappa, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.2)
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.set_label(r"$\mathsf{\kappa}$", rotation=0, fontsize=70)#, labelpad=5)
    cb2.ax.tick_params(labelsize=60)

    # Terzo subplot: Signal-to-Noise Ratio
    ax3 = plt.subplot(1, 3, 3, aspect='equal')
    ax3.set_xlabel(r"$\mathrm{R.A. \; [arcmin]}$", fontsize=70)
    ax3.set_ylabel(r"$\mathrm{Dec. \; [arcmin]}$", fontsize=70)
    ax3.set_title("Signal-to-Noise Ratio", fontsize=70)
    ax3.tick_params(axis='both', labelsize=60)
    im3 = ax3.imshow(SNR, origin='lower', extent=extent, vmin=-4, vmax=4, cmap=cmap)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.2)
    cb3 = plt.colorbar(im3, cax=cax3)
    cb3.set_label(r"$\mathrm{SNR}$", rotation=270, fontsize=70, labelpad=20)
    cb3.ax.tick_params(labelsize=60)

    plt.tight_layout()
    
def plot_map(fov_arcmin, mapp, map_type):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    vmin = -0.0247215488
    vmax = 0.0247215488
    ranges = fov_arcmin / 2.0
    extent = [-ranges, ranges, -ranges, ranges]
    cmap = cm.coolwarm

    fig, ax = plt.subplots(1, 1, figsize=(16.5, 13.8))
    ax.set_aspect('equal')
    ax.tick_params(labelsize=40)
    plt.xlabel('$\mathrm{R. A. \;\;[arcmin]}$', fontsize=40, labelpad=20)
    plt.ylabel('$\mathrm{Dec. \;\;[arcmin]}$', fontsize=40, labelpad=5)
    plt.xlim(-ranges, ranges)
    plt.ylim(-ranges, ranges)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)

    # Visualizza l'immagine
    if map_type == "kappa":
        im = ax.imshow(mapp, origin='lower', extent=extent, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"$\mathsf{\kappa}$", rotation=0, fontsize=40, labelpad=5)
        cb.ax.tick_params(labelsize=40)
    elif map_type == "SNR":
        im = ax.imshow(mapp, origin='lower', extent=extent, aspect="equal", cmap=cmap, vmin=-4, vmax=4)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(r"$\mathrm{SNR}$", rotation=270, fontsize=40, labelpad=5)
        cb.ax.tick_params(labelsize=40)

    #plt.tight_layout()
    
def compute_PS(input_map,FieldSize,nbins=128,lrmin=-2):
    """
    Compute the angular power spectrum of input_map.
    :param input_map: input map (n x n numpy array)
    :param FieldSize: the side-length of the input map in degrees
    :return: l, Pl - the power-spectrum at l
    """
    # set the number of pixels and the unit conversion factor
    npix = input_map.shape[0]
    factor = 2.0*np.pi/(FieldSize*np.pi/180.0)

    # take the Fourier transform of the input map:
    fourier_map = fftengine.fftn(input_map)/npix**2
    
    # compute the Fourier amplitudes
    fourier_amplitudes = np.abs(fourier_map)**2
    fourier_amplitudes = fourier_amplitudes.flatten()

    # compute the wave vectors
    kfreq = fftengine.fftfreq(input_map.shape[0])*input_map.shape[0]
    kfreq2D = np.meshgrid(kfreq, kfreq)
    # take the norm of the wave vectors
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()

    # set up k bins. The PS will be evaluated in these bins
    half = npix/2
    rbins = int(np.sqrt(2*half**2))+1
    #kbins = np.linspace(0.0,rbins,(rbins+1))
    #print(kbins,rbins)
    kbins = np.linspace(lrmin,np.log10(rbins),nbins)
    kbins = 10**kbins
    #print(kbins)
    #return 0
    
    # use the middle points in each bin to define the values of k where the PS is evaluated
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*factor

    # now compute the PS: calculate the mean of the
    # Fourier amplitudes in each kbin
    Pbins, _, _ = scipy.stats.binned_statistic(knrm,fourier_amplitudes,statistic = "mean",bins = kbins)
    # return kvals and PS
    l=kvals[1:]
    Pl=Pbins[1:]/factor**2
    return l, Pl

def lcorr(theta):
    theta = theta / 60.0 * np.pi/180.0
    return 2*np.pi/(theta)

def oursort(e):
    thresholdString = e.replace(e[-4:],"")
    if (thresholdString[-2] != "."): 
        thresholdString = thresholdString + ".0"        
    key = thresholdString[thresholdString.rfind("_")+1:]
    key = int(key.replace(".",""))
    return key

def find_r_bel_rec(x_c, y_c, raggio, x, y, n, i, r_bel, r_x, r_y):
    
    if n == len(x):
        return r_bel, r_x, r_y
    if x_c.size == 1:
        distx = x[n] - x_c
        disty = y[n] - y_c
    else:
        distx = x[n] - x_c[i]
        disty = y[n] - y_c[i]
    
    dist = np.sqrt(distx**2 + disty**2)
    
    if ((dist < raggio[n]) and (raggio[n] > 0)):
        if not any(np.isclose(raggio[n], r) for r in r_bel[i]):
            r_bel[i].append(raggio[n])
            r_x[i].append(x[n])
            r_y[i].append(y[n])
    
    return find_r_bel_rec(x_c, y_c, raggio, x, y, n+1, i, r_bel, r_x, r_y)

def divide_found_circles(found_circles):
    x_values = []
    y_values = []
    raggio_values = []

    for circle in found_circles:
        x_values.append(circle[0])
        y_values.append(circle[1])
        raggio_values.append(circle[2])

    return x_values, y_values, raggio_values


def radial_profile(data, center):
    """
    Calculate the radial profile of a map defined the
    center, results are done in pixel scales
    """
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), data.ravel())
    #rbin = np.bincount(r.ravel(), r.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    #r_ = rbin/nr
    return radialprofile

def pop_sources(clid, center, fov, sources_density):
    """
    Populate the field of view of the map and sample it
    with point sources with a given density in square arcmin
    fov in deg
    clid the seed of the random number generator
    """
    arcsec = fov*3600.0
    arcmin2 = arcsec*arcsec/60.0/60.0
    nsources = sources_density*arcmin2
    nsources = int(nsources)
    t = np.zeros([nsources])
    np.random.seed(clid)
    xs = np.random.uniform(-0.5,0.5,nsources)
    ys = np.random.uniform(-0.5,0.5,nsources)
    xs = xs*arcsec/60.0
    ys = ys*arcsec/60.0
    
    xs_ = xs - center[0]
    ys_ = ys - center[1]
  
    t = np.arctan(xs_/ys_)
    
    #print('   - - -  ')
    #print('   Sampling the fov with point like sources  ')
    #print('   - - -  ')
    #print('   - - -  ')
    #print(' min and max values of the sources in the fov [-0.5,0,5] x arcsec ')
    #print(np.amin(xs), np.amax(xs))
    #print(np.amin(ys), np.amax(ys))
    #print('   - - -  ')
    rs = np.sqrt(xs_*xs_ + ys_*ys_ )
    #print( ' min and max values from the center')
    #print(np.amin(rs), np.amax(rs))
    #print('   - - -  ')
    return xs, ys, t, rs

def area_of_polygon(x, y):
    """
    Calculates the area of an arbitrary polygon given its verticies
    """
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0

def getInfo(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def perimeter_of_polygon(x, y):
    N = len(x)
    firstx = x[0]
    firsty = y[0]
    prevx, prevy = firstx, firsty
    res = 0

    for i in range(1, N):
        nextx = x[i]
        nexty = y[i]
        res = res + getInfo(prevx,prevy,nextx,nexty)
        prevx = nextx
        prevy = nexty
    res = res + getInfo(prevx,prevy,firstx,firsty)
    return res

def find_peaks(folderout, SNR, npixel, fov_arcmin, threshold, kappa, str_SNRflag, i_fil, case=1):
    """
    case = -1 finds valleys
    """
    if case<0:
        SNR=-SNR
    
    warnings.filterwarnings('ignore')
    #ranges = fov_arcmin/2.0
    #plt.rcParams['figure.figsize'] = [33, 13]
    #plt.subplot(1,2,1)
    #plt.axis('equal')
    #matplotlib.rc('xtick', labelsize=15)
    #matplotlib.rc('ytick', labelsize=15)
    #plt.xlabel('$\mathsf{R. A. \;\;[arcmin]}$', fontsize=16)
    #plt.ylabel('$\mathsf{Dec. \;\;[arcmin]}$', fontsize=16)
    #plt.title("summary image peaks",fontsize=16)
    #extent = [-ranges,ranges, -ranges, ranges]

    #<--- MASKS FOR PEAKS above the SNR threshold --->
    peaks =  SNR >=threshold

    # perform connected component analysis
    labeled_image, count = skimage.measure.label(peaks, return_num=True, connectivity=2)
    print(" number of connected regions SNR>=", threshold, ': ', count)
    
    #gray_scale = skimage.color.rgb2gray(SNR)
    gray_scale = SNR
    summary_image = skimage.color.gray2rgb(gray_scale)

    # color each of the colonies a different color
    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    summary_image[peaks] = colored_label_image[peaks]

    #plt.imshow(summary_image,origin='lower',extent=extent,aspect="auto",cmap='gray')#,vmin=-1e-2,vmax=1e-2)
    #cb = plt.colorbar()
    #cb.set_label('$\mathsf{labeled \; images}}$',rotation=270,fontsize=16)

    object_features = skimage.measure.regionprops(labeled_image,SNR)
    object_areas = [objf["area"] for objf in object_features]
    object_perimeters = [objf["perimeter"] for objf in object_features]
    object_im = [objf["intensity_mean"] for objf in object_features]
    object_imax = [objf["intensity_max"] for objf in object_features]
    object_centroid_w = [objf["centroid_weighted"] for objf in object_features]
    object_centroid_wl = [objf["centroid_weighted_local"] for objf in object_features]
    object_en = [objf["euler_number"] for objf in object_features]
    object_ecc = [objf["eccentricity"] for objf in object_features]
    object_solidity = [objf["solidity"] for objf in object_features]
    object_majorL = [objf["axis_major_length"] for objf in object_features]
    object_minorL = [objf["axis_minor_length"] for objf in object_features]
    object_label = [objf["label"] for objf in object_features]
    object_areas_convex = [objf["area_convex"] for objf in object_features]
    object_coors = [objf["coords"] for objf in object_features]
    object_areas_bbox = [objf["area_bbox"] for objf in object_features]
    object_feret_dmax = [objf["feret_diameter_max"] for objf in object_features]
    object_minorL =np.array(object_minorL)
    object_majorL =np.array(object_majorL)
    ell_area = np.pi*object_minorL*object_majorL
    ell_min_over_maj = 1.0 - object_minorL/object_majorL 
    ell_min_over_maj = np.nan_to_num(ell_min_over_maj, nan=-1)

    xc = []
    yc = []
    xc_max = []
    yc_max = []
    
    for i in range(0,len(object_centroid_w)):
        xc_, yc_ = object_centroid_w[i]
        xc = np.append(xc,xc_)
        yc = np.append(yc,yc_)
        coor = np.array(object_coors[i])
        _SNR = np.zeros(SNR.shape)
        _SNR[coor[:,0],coor[:,1]]=SNR[coor[:,0],coor[:,1]] 
        max_x,max_y =np.where(_SNR==_SNR.max())
        xc_max = np.append(xc_max,max_x)
        yc_max = np.append(yc_max,max_y)
    
#...convert in arcmin...

    object_areas = np.array(object_areas)
    object_areas_convex = np.array(object_areas_convex)
    object_areas_bbox = np.array(object_areas_bbox)
    object_perimeters = np.array(object_perimeters)
    object_feret_dmax = np.array(object_feret_dmax)
    object_areas = object_areas.astype(np.float32)
    object_areas_convex = object_areas_convex.astype(np.float32)
    object_areas_bbox = object_areas_bbox.astype(np.float32)
    object_areas = object_areas*(fov_arcmin/npixel)**2
    object_areas_convex = object_areas_convex*(fov_arcmin/npixel)**2
    object_areas_bbox = object_areas_bbox*(fov_arcmin/npixel)**2
    ell_area = ell_area*(fov_arcmin/npixel)**2
    object_perimeters = object_perimeters*(fov_arcmin/npixel)
    object_feret_dmax = object_feret_dmax*(fov_arcmin/npixel)
    
    filout = None
    
    if len(xc) > 0:
        xc = (xc-npixel/2)*(fov_arcmin/npixel)
        yc = (yc-npixel/2)*(fov_arcmin/npixel)

        xc_max = (xc_max-npixel/2)*(fov_arcmin/npixel)
        yc_max = (yc_max-npixel/2)*(fov_arcmin/npixel)
    else:
        if case > 0:
            dir_out = os.path.join(folderout, 'peaks_' + i_fil + 'k')
            filout = os.path.join(dir_out, 'peaks_SNR'+str_SNRflag+'_'+str(threshold)+'.txt')
        else:
            dir_out = os.path.join(folderout, 'valleys_' + i_fil + 'k')
            filout = os.path.join(dir_out, 'valleys_SNR'+str_SNRflag+'_'+str(threshold)+'.txt')

            empty_data = np.zeros((1, 16))
            
            header="LABEL  AREA[arcmin2]  PERIMETER[arcmin]  XCM[arcmin]  YCM[arcmin]  MAX_SNR  MEAN_SNR  ECCENTRICITY  SOLIDITY  ELLIPTICITY  AREA_CONVEX[armin2]  X_MAX[arcmin]  Y_MAX[arcmin]  AREA_BBOX[arcmin2]  AREA_ELL[arcmin2]  FeretDmax[arcmin]"

            np.savetxt(filout, empty_data,
                       fmt='%i   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f  %f  %f  %f  ',
                       header=header,
                       comments='')        

    if case>0:
        dir_out = os.path.join(folderout, 'peaks_' + i_fil + 'k')
        filout = os.path.join(dir_out, 'peaks_SNR'+str_SNRflag+'_'+str(threshold)+'.txt')
    else:
        dir_out = os.path.join(folderout, 'valleys_' + i_fil + 'k')
        filout = os.path.join(dir_out, 'valleys_SNR'+str_SNRflag+'_'+str(threshold)+'.txt')
        
        object_imax = - np.array(object_imax)
        object_im = - np.array(object_im)
    header="LABEL  AREA[arcmin2]  PERIMETER[arcmin]  XCM[arcmin]  YCM[arcmin]  MAX_SNR  MEAN_SNR  ECCENTRICITY  SOLIDITY  ELLIPTICITY  AREA_CONVEX[armin2]  X_MAX[arcmin]  Y_MAX[arcmin]  AREA_BBOX[arcmin2]  AREA_ELL[arcmin2]  FeretDmax[arcmin]"
    np.savetxt(filout, np.c_[object_label, object_areas, object_perimeters, xc, yc, object_imax, object_im, object_ecc,\
    object_solidity, ell_min_over_maj, object_areas_convex, xc_max, yc_max, object_areas_bbox, ell_area, object_feret_dmax],
               fmt='%i   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f  %f  %f  %f  ',
               header=header,
               comments='')

def clean(filin_cat, voids, plot_results=0, final_map=0):
    
    object_label,object_areas,object_perimeters,xc,yc,object_imax,object_im,object_ecc,\
    object_solidity,ell_min_over_maj,object_areas_convex,xc_max,yc_max,object_areas_bbox,\
    ell_area,object_feret_dmax = np.loadtxt(filin_cat, unpack=True, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], skiprows=1)
    rc = -np.sqrt(object_areas/np.pi)
    n = np.size(object_label)
    #print('number of elements:', n)
    object_label = object_label.astype(int)
    #...sort them in radius...
    if n == 1:
        points = [(rc, object_label, object_areas, object_perimeters, xc, yc, object_imax, object_im, object_ecc,
                   object_solidity, ell_min_over_maj, object_areas_convex, xc_max, yc_max, object_areas_bbox,
                   ell_area, object_feret_dmax)]
    else:
        points = zip(rc, object_label, object_areas, object_perimeters, xc, yc, object_imax, object_im, object_ecc,
                     object_solidity, ell_min_over_maj, object_areas_convex, xc_max, yc_max, object_areas_bbox,
                     ell_area, object_feret_dmax)
    sorted_points = sorted(points)
    
    rc  = np.array([point[0] for point in sorted_points])
    object_label = np.array([point[1] for point in sorted_points])
    object_areas = np.array([point[2] for point in sorted_points])
    object_perimeters = np.array([point[3] for point in sorted_points])
    xc = np.array([point[4] for point in sorted_points])
    yc = np.array([point[5] for point in sorted_points])
    object_imax = np.array([point[6] for point in sorted_points])
    object_im = np.array([point[7] for point in sorted_points])
    object_ecc = np.array([point[8] for point in sorted_points])
    object_solidity = np.array([point[9] for point in sorted_points])
    ell_min_over_maj = np.array([point[10] for point in sorted_points])
    object_areas_convex = np.array([point[11] for point in sorted_points])
    xc_max = np.array([point[12] for point in sorted_points])
    yc_max = np.array([point[13] for point in sorted_points])
    object_areas_bbox = np.array([point[14] for point in sorted_points])
    ell_area = np.array([point[15] for point in sorted_points])
    object_feret_dmax = np.array([point[16] for point in sorted_points])
    rc = -rc
    
    for i in range(0,n):
        if voids == 1:
            dist = np.sqrt((xc-xc[np.abs(object_label[i])-1])**2 + (yc-yc[np.abs(object_label[i])-1])**2)
        else:
            dist = np.sqrt((xc_max-xc_max[np.abs(object_label[i])-1])**2 + (yc_max-yc_max[np.abs(object_label[i])-1])**2)
        rr = 0.5*(rc+rc[np.abs(object_label[i])-1])
        merge = dist < rr
        if object_label[i]>0:
            h = np.where(merge)
            h = np.asarray(h[0]) + 1
            if(np.size(h)>1):
                object_label[h[1:]-1] = -object_label[h[0]-1]
    filout_cat = filin_cat
    header="LABEL  AREA[arcmin2]  PERIMETER[arcmin]  XCM[arcmin]  YCM[arcmin]  MAX_SNR  MEAN_SNR  ECCENTRICITY  SOLIDITY  ELLIPTICITY  AREA_CONVEX[armin2]  X_MAX[arcmin]  Y_MAX[arcmin]  AREA_BBOX[arcmin2]  AREA_ELL[arcmin2]  FeretDmax[arcmin]"
    np.savetxt(filout_cat, np.c_[object_label, object_areas, object_perimeters, xc, yc, object_imax, object_im, object_ecc,\
                                object_solidity, ell_min_over_maj, object_areas_convex, xc_max, yc_max, object_areas_bbox,\
                                ell_area,object_feret_dmax],
               fmt='%i   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f   %f  %f  %f  %f  ',
               header=header,
               comments='')
    
def read_peaks(filin,smooth):
    snr = []
    npk = []
    area = []
    area_convex = []
    for i in range(-1,10):
        filinpk = filin + "peaks_SNR"+str(smooth)+"_"+str(i+1)+".txt"
        idh,a,ac = np.loadtxt(filinpk, unpack=True, usecols=[0,1,10], skiprows=1)
        a = a [idh>0]
        ac = ac [idh>0]
        snr = np.append(snr,i+1)
        npk = np.append(npk,np.size(a))
        area = np.append(area,a)
        area_convex = np.append(area_convex,ac)
        
        if i<9:
            filinpk = filin + "peaks_SNR"+str(smooth)+"_"+str(i+1.5)+".txt"
            idh,a,ac = np.loadtxt(filinpk, unpack=True, usecols=[0,1,10], skiprows=1)
            a = a [idh>0]
            ac = ac [idh>0]
            snr = np.append(snr,i+1.5)
            npk = np.append(npk,np.size(a))
            area = np.append(area,a)
            area_convex = np.append(area_convex,ac)
    return snr,npk, area, area_convex

def read_valleys(filin, smooth):
    snr = []
    npk = []
    area = []
    area_convex = []
    for i in range(-1,4):
        filinpk = filin + "valleys_SNR"+str(smooth)+"_"+str(i+1)+".txt"
        idh,a,ac = np.loadtxt(filinpk, unpack=True, usecols=[0,1,10], skiprows=1)
        a = a [idh>0]
        ac = ac [idh>0]
        snr = np.append(snr,i+1)
        npk = np.append(npk,np.size(a))
        area = np.append(area,a)
        area_convex = np.append(area_convex,ac)
        if i<3:
            filinpk = filin + "valleys_SNR"+str(smooth)+"_"+str(i+1.5)+".txt"
            idh,a,ac = np.loadtxt(filinpk, unpack=True, usecols=[0,1,10], skiprows=1)
            a = a [idh>0]
            ac = ac [idh>0]
            snr = np.append(snr,i+1.5)
            npk = np.append(npk,np.size(a))
            area = np.append(area,a)
            area_convex = np.append(area_convex,ac)
    return snr,npk, area, area_convex

def map_PDF_and_Pl(folderout, SNR_final, kappa_original, kappa_final, fov_arcmin, smooth_filter_set, points_PDF, points_Pl):
    import Finder_functions as mystery
    
    ##PDF(SNR)
    plt.figure(figsize=(6, 6))
    #plt.xlim(-10,10)
    #plt.ylim(7e-4,5e-1)
    #plt.yticks(fontsize=15)
    #plt.xticks(fontsize=15)
    #plt.yscale('log')
    #plt.xlabel('SNR', fontsize=25)
    #plt.ylabel('PDF', fontsize=25)
    
    SNR_binning = np.linspace(-10, 30, points_PDF)
    counts_SNR, bins_SNR, p = plt.hist(SNR_final.flatten(), bins=SNR_binning, density=True, color='white')
    bins_SNR = ((bins_SNR[1:] + bins_SNR[:-1])/2)
    #plt.plot(bins_SNR, counts_SNR,linestyle='-', color='tab:green')

    ##PDF(K)
    #plt.figure(figsize=(6, 6))
    #plt.xlim(-2, 2)
    #plt.ylim(5e-2, 5e1)
    plt.ylim(5e-2, 7e1)
    plt.xlim(-1, 1)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yscale('log')
    plt.xlabel('$\mathsf{\kappa} \, [10^{{-1}}]$', fontsize=25)
    plt.ylabel('PDF($\mathsf{\kappa}$)',fontsize=25)
    
    K_binning = np.linspace(-0.1, 1, points_PDF)
    counts_K, bins_K, p = plt.hist(kappa_original.flatten(), bins=K_binning, density=True, color='white')
    bins_K = ((bins_K[1:] + bins_K[:-1])/2)
    counts_KK, _, _ = plt.hist(kappa_final.flatten(), bins=K_binning, density=True, color='white')
    plt.plot(bins_K*10, counts_K, linestyle='-', color='black')
    plt.plot(bins_K*10, counts_KK, linestyle='-', color='tab:blue')

    filout_PDF = folderout + "PDF.txt"
    np.savetxt(filout_PDF, np.c_[bins_K, counts_K, counts_KK, bins_SNR, counts_SNR],
               fmt='%f     %f     %f     %f     %f',
               header="bins_k   counts_K_original   counts_K_final   bins_SNR   counts_SNR",
               comments='')   

    ##P(K)
    l, p_original = mystery.compute_PS(kappa_original, fov_arcmin/60.0, points_Pl)
    l = l[p_original>0]
    p_original = p_original[p_original>0]

    _, p_final = mystery.compute_PS(kappa_final, fov_arcmin/60.0, points_Pl)
    #l_final = l_final[p_final>0]
    p_final = p_final[p_final>0]
    gl_final = mystery.lcorr(smooth_filter_set)/2/np.pi
    g = np.linspace(1e-6, 1e-4, 3)
    pk_peak = g/g*gl_final

    plt.figure(figsize=(6, 6))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$l$', fontsize=25)
    plt.ylabel(r'$l^2 P_{\mathsf{\kappa}}(l)$', fontsize=25)
    plt.xlim(1e2,1e4)
    plt.ylim(1e-6,1e-4)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.plot(l, p_original*l**2, label='noiseless', color='black')
    smooth_filter_set = str(smooth_filter_set)
    plt.plot(l, p_final*l**2, label=f'with shape noise and $\\theta_G={smooth_filter_set}$ arcmin')
    plt.plot(pk_peak, g, color='tab:blue', linestyle=":")
    plt.legend(fontsize=15)

    filout_Pl = folderout + "Pl.txt"
    np.savetxt(filout_Pl, np.c_[l, p_original, p_final],
               fmt='%f     %e     %e',
               header="l   P(K_original)   P(K_final)",
               comments='')   

def connected_regions(filin, folderout, ell_err, smooth_filter_set, n_gal_set, path, p_obj, i_fil, n_seed=1234):
    import Finder_functions as mystery
    final_kappa, kappa, fov_arcmin, sigma_noise, npixel, zs = mystery.noise_and_smooth(filin, ell_err, smooth_filter_set, n_gal_set, n_seed)
    SNR = final_kappa / sigma_noise

    for i in range(-1, 4):
        mystery.find_peaks(folderout, SNR, npixel, fov_arcmin, i + 1, kappa, str(smooth_filter_set), i_fil, -1)    
        if i < 3:
            mystery.find_peaks(folderout, SNR, npixel, fov_arcmin, i + 1.5, kappa, str(smooth_filter_set), i_fil, -1)

    filin_cat_list = glob.glob(path + p_obj + "_SNR" + str(smooth_filter_set) + '*.txt')
    filin_cat_list.sort(key=mystery.oursort)

    for filin_cat in filin_cat_list:
        #print(filin_cat)
        voids = 0
        mystery.clean(filin_cat, voids)
    
    return SNR, final_kappa, kappa, fov_arcmin, npixel
        
def void_minima(path, p_obj, smooth_filter_set, SNR, fov_arcmin, threshold_value=0, stop_th=0):
    
    #ranges = fov_arcmin/2
    #plt.rcParams['figure.figsize'] = [33, 13]
    #plt.subplot(1,2,1)
    #plt.axis('equal')
    #margin = 25
    #plt.ylim(-ranges-margin,ranges+margin)
    #plt.xlim(-ranges-margin,ranges+margin)
    #matplotlib.rc('xtick', labelsize=15)
    #matplotlib.rc('ytick', labelsize=15)
    #plt.xlabel('$\mathsf{R. A. \;\;[arcmin]}$', fontsize=16)
    #plt.ylabel('$\mathsf{Dec. \;\;[arcmin]}$', fontsize=16)
    #extent = [-ranges+margin, ranges-margin, -ranges+margin, ranges-margin]
    
    import Finder_functions as mystery
    
    txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
    txt_files = [file for file in txt_files if f"{smooth_filter_set}_" in file and file.endswith(".txt")]
    txt_files.sort(key=lambda x: float(x.split('_')[-1][:-4]) if f"{smooth_filter_set}_" in x else -1)

    num_files = len(txt_files)
    i = 0
    j = 0
    r_v = []
    r_v_t = []
    max_voids = 0
    etichette = []
    # etichette_valide = []

    for file in txt_files:
        if float(file.split('_')[-1][:-4]) <= threshold_value:
            i += 1
            num_voids = np.loadtxt(os.path.join(path, file), skiprows=1).shape[0]
            if num_voids > max_voids:
                max_voids = num_voids
        else:
            etichette.append(None)

    for i, file in enumerate(txt_files):
        data = np.loadtxt(os.path.join(path, file), skiprows=1)

        if len(data) == 0:
            r_v_t.append([np.nan] * max_voids)
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        num_voids = len(data)
        r_v = np.zeros(max_voids)

        for j in range(num_voids):
            void = data[j]
            if void.ndim == 1:
                void = void.reshape(1, -1)

            y_c = void[0][11]
            x_c = void[0][12]
            area = void[0][1]
            raggio = np.sqrt(area / np.pi)

            if j >= len(r_v):
                r_v = np.append(r_v, np.zeros(j - len(r_v) + 1))

            r_v[j] = raggio

            # if etichette[i] is not None:
            # colore = etichette[i]['colore']
            # plt.plot(x_c, y_c, marker='.', color=colore, linestyle='')
            # cerchio = patches.Circle((x_c, y_c), raggio, edgecolor=colore, facecolor=colore)
            # ax = plt.gca()
            # ax.add_patch(cerchio)
        r_v_t.append(r_v)

    file_name = f'{p_obj}_SNR{str(smooth_filter_set)}_{str(threshold_value)}.txt'
    index = txt_files.index(file_name)

    if np.all(np.isnan(r_v_t[-1])):
        exist = False
        print("No voids found")
    else:
        exist = True
        r_v = r_v_t[index]
        a_c, x_max, y_max = np.around(np.genfromtxt(path + file_name, usecols=(1, 11, 12), skip_header=1), decimals=8).T
        x_max, y_max = y_max, x_max

        n_file_s = 1
        range_file = num_files - n_file_s
        color_index = range_file - 1
        ct_count = 0
        c_count = 0
        c_m_count = 0

        soglia = (threshold_value - stop_th)*2

        r_bel = [[] for _ in range(range_file)] 
        r_x = [[] for _ in range(range_file)]
        r_y = [[] for _ in range(range_file)]

        unique_xf = []
        unique_yf = []
        unique_rf = []
        found_circles = []
        seen_values = set()

        for i in range(range_file):
            # colore = colori[color_index]
            # color_index -= 1
            # if color_index < 0:
            #     color_index = len(colori) - 1

            if int(threshold_value - (n_file_s * 0.5) - (i * 0.5)) == (threshold_value - (n_file_s * 0.5) - (i * 0.5)):
                _th = str(int(threshold_value - (n_file_s * 0.5) - (i * 0.5)))

            else:
                _th = str(threshold_value - (n_file_s * 0.5) - (i * 0.5))

            if float(_th) < stop_th:
                break
            file_name = f'{p_obj}_SNR{str(smooth_filter_set)}_{_th}.txt'
            x_m = np.zeros(max_voids)
            y_m = np.zeros(max_voids)
            raggio = np.zeros(max_voids)
            a, b, c, m, n = np.loadtxt(path + file_name, unpack=True, usecols=[1, 3, 4, 11, 12], skiprows=1)

            for k in range(len(a)):
                x_m[k] = n[k]
                y_m[k] = m[k]
                raggio[k] = np.sqrt(a[k] / np.pi)

            r_bel = [[] for _ in range(x_max.size)] 
            r_x = [[] for _ in range(x_max.size)]
            r_y = [[] for _ in range(x_max.size)]

            for n in range(len(x_m)):
                # plt.plot(x_m[n], y_m[n], marker='.', color=colore, linestyle='')
                ct_count += 1
            for ii in range(x_max.size):
                r_values_max, x_m_values, y_m_values = mystery.find_r_bel_rec(x_max, y_max, raggio, x_m, y_m, 0, ii, r_bel, r_x, r_y)
                for xd in range(len(r_values_max[ii])):
                    if (x_m_values[ii][xd], y_m_values[ii][xd], r_values_max[ii][xd]) not in seen_values:

                        if i == int(soglia - 1):
                            if (x_m_values[ii][xd], y_m_values[ii][xd]) in zip(unique_xf, unique_yf):
                                unique_circles = zip(unique_xf, unique_yf, unique_rf)
                                current_circle = (x_m_values[ii][xd], y_m_values[ii][xd], r_values_max[ii][xd])
                                if current_circle in found_circles:
                                    continue

                                for unique_circle in unique_circles:
                                    if current_circle[:2] == unique_circle[:2] and current_circle[2] > unique_circle[2]:
                                        found_circles.append(current_circle)
                                        c_m_count += 1
                                        break                                
                            elif soglia == 1:
                                unique_circles = [(x_m_values[ii][xd], y_m_values[ii][xd], r_values_max[ii][xd])]
                                current_circle = (x_m_values[ii][xd], y_m_values[ii][xd], r_values_max[ii][xd])
                                found_circles.append(current_circle)
                                c_m_count += 1

                        else:
                            unique_xf.append(x_m_values[ii][xd])
                            unique_yf.append(y_m_values[ii][xd])
                            unique_rf.append(r_values_max[ii][xd])
                            seen_values.add((x_m_values[ii][xd], y_m_values[ii][xd], r_values_max[ii][xd]))

                    # if r_values_max[ii][xd] > 0:
                        # plt.plot(x_m_values[ii][xd], y_m_values[ii][xd], marker='x', color='green', linestyle='', markersize=10) #i centri di massa dei vuoti che cadono all'interno di dist < raggio[n]
                        # cerchio = patches.Circle((x_m_values[ii][xd], y_m_values[ii][xd]), r_values_max[ii][xd], edgecolor=colore, facecolor='none')
                        # ax = plt.gca()
                        # ax.add_patch(cerchio)

        # plt.plot(x_max, y_max, marker='.',color='k',linestyle='')
        # print("Total number of void SNR minima among the analyzed thresholds:", ct_count)
        #print(f"Number of void SNR minima with threshold {threshold_value}: {x_max.size}")
        #print("Total number of watershed void SNR minima:", c_m_count)

        # legend_handles = []
        # for etichetta in etichette_valide:
        #     line = Line2D([0], [0], linestyle='-', linewidth=8, color=etichetta['colore'])
        #     legend_handles.append(line)

        # plt.legend(legend_handles, [etichetta['nome'] for etichetta in etichette_valide], fontsize=20)
        # plt.show()
        
        SNR = -SNR
        map = np.copy(SNR)
        snr_map = map
        unique_xf_m, unique_yf_m, unique_rf_m = mystery.divide_found_circles(found_circles)
        x_min = []
        y_min = []

        for i in range(len(unique_xf_m)):
            d_m = None
            x_min.append([i])
            y_min.append([i])
            for n in range(x_max.size):
                if x_max.size == 1:
                    distx_f = x_max - unique_xf_m[i] 
                    disty_f = y_max - unique_yf_m[i]
                else:
                    distx_f = x_max[n] - unique_xf_m[i] 
                    disty_f = y_max[n] - unique_yf_m[i] 
                dist_f = np.sqrt(distx_f**2 + disty_f**2)
                if ((d_m is None) or (dist_f < d_m)):
                    d_m = dist_f
                    if(d_m != None):
                        if x_max.size == 1:
                            x_min = x_max
                            y_min = y_max
                        else:
                            x_min[i] = x_max[n]
                            y_min[i] = y_max[n]

        x_v_arc = np.array(x_min)
        y_v_arc = np.array(y_min)

        
        return exist, x_v_arc, y_v_arc, snr_map

def void_radii(exist, folderout, snr_map, fov_arcmin, threshold_value, x_v_arc, y_v_arc, npixel, SNR_threshold):
    if exist:
        ranges = fov_arcmin/2
        x_v_pix = np.int32((x_v_arc) / fov_arcmin * npixel + npixel * 0.5)
        y_v_pix = np.int32((y_v_arc) / fov_arcmin * npixel + npixel * 0.5)

        circle_distance_threshold = 0.5
        voids_output = np.zeros((x_v_pix.size, 4))
        voids_output[:, 0] = x_v_arc
        voids_output[:, 1] = y_v_arc
        voids_output_over = np.zeros((x_v_pix.size, 2))
        x_inter = np.zeros((x_v_pix.size))
        all_radii = []
        all_avg_snr = []

        for i in range(x_v_pix.size):
            if x_v_pix.size == 1:
                center_x = int(x_v_pix)
                center_y = int(y_v_pix)
            else:
                center_x = int(x_v_pix[i])
                center_y = int(y_v_pix[i])
            max_radius = 1
            grid_size = 3
            edges_x = center_x - max_radius
            edges_xx = center_x + max_radius
            edges_y = center_y - max_radius
            edges_yy = center_y + max_radius
            x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            mask = ((x - max_radius - 0.5) ** 2 + (y - max_radius - 0.5) ** 2) <= (
                    max_radius + circle_distance_threshold) ** 2
            avg_snr = 5
            radii = []
            avg_snr_values = []

            while avg_snr >= SNR_threshold:
                grid_size = 2 * max_radius + 1
                void_pixels = snr_map[
                    center_y - max_radius:center_y + max_radius + 1, center_x - max_radius:center_x + max_radius + 1]

                x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
                mask = ((x - max_radius - 0.5) ** 2 + (y - max_radius - 0.5) ** 2) <= (
                            max_radius + circle_distance_threshold) ** 2
                mask = np.pad(mask, ((0, void_pixels.ndim - mask.ndim),) * 2, mode='constant',
                              constant_values=np.nan)[:void_pixels.shape[0], :void_pixels.shape[1]]

                if center_y - max_radius >= 0 and center_x - max_radius >= 0 and center_y + max_radius + 1 <= snr_map.shape[
                    0] and center_x + max_radius + 1 <= snr_map.shape[1]:
                    void_pixels = snr_map[center_y - max_radius:center_y + max_radius + 1,
                                  center_x - max_radius:center_x + max_radius + 1]

                    void_pixels[mask]
                    avg_snr = np.nanmean(void_pixels)

                else:
                    edges_x = max(center_x - max_radius, 0)
                    edges_xx = min(center_x + max_radius + 1, snr_map.shape[1])
                    edges_y = max(center_y - max_radius, 0)
                    edges_yy = min(center_y + max_radius + 1, snr_map.shape[0])
                    void_pixels = snr_map[edges_y:edges_yy, edges_x:edges_xx]

                    void_pixels[mask]
                    avg_snr = np.nanmean(void_pixels)

                max_radius += 1

                radius = max_radius - 1 + circle_distance_threshold

                voids_output_over[i][0] = (radius - npixel / 2) * (fov_arcmin / npixel) + fov_arcmin / 2
                voids_output_over[i][1] = avg_snr

                radii.append((radius - npixel / 2) * (fov_arcmin / npixel) + fov_arcmin / 2)
                avg_snr_values.append(avg_snr)

                if (avg_snr >= SNR_threshold):
                    voids_output[i][2] = (radius - npixel / 2) * (fov_arcmin / npixel) + fov_arcmin / 2
                    voids_output[i][3] = avg_snr

            x_inter[i] = np.interp(SNR_threshold, [voids_output_over[i][1], voids_output[i][3]],
                                    [voids_output_over[i][0], voids_output[i][2]])

            all_radii.append(radii)
            all_avg_snr.append(avg_snr_values)
            # print(i, all_radii, all_avg_snr)

        ############## CLEANER ###########################################################
        sorted_indices = sorted(range(len(x_inter)), key=lambda k: x_inter[k], reverse=True)

        if len(sorted_indices) == 1:
            x_inter_s = x_inter
            x_v_arc_s = x_v_arc
            y_v_arc_s = y_v_arc
            n = 1
        else:
            x_inter_s = [x_inter[i] for i in sorted_indices]
            x_v_arc_s = [x_v_arc[i] for i in sorted_indices]
            y_v_arc_s = [y_v_arc[i] for i in sorted_indices]
            n = len(x_v_arc_s)

        exclude_flags = [False] * n

        for c in range(n):
            if exclude_flags[c]:
                continue

            for f in range(c + 1, n):
                if exclude_flags[f]:
                    continue

                d = np.sqrt(((x_v_arc_s[c] - x_v_arc_s[f]) ** 2) + ((y_v_arc_s[c] - y_v_arc_s[f]) ** 2))
                if d == 0:
                    continue

                s = (x_inter_s[c] + x_inter_s[f]) * 0.75
                r_sup = max(x_inter_s[c], x_inter_s[f])

                if d <= s or d <= r_sup:
                    if x_inter_s[c] < x_inter_s[f]:
                        exclude_flags[c] = True
                    else:
                        exclude_flags[f] = True

        if n == 1:
            x_arc_f = x_v_arc_s
            y_arc_f = y_v_arc_s
            r_arc_f = x_inter_s
            print("Voids found:", x_v_arc.size)
            print("Voids after cleaning:", x_arc_f.size)

            # FINAL PLOT and SAVING
            # plt.plot(x_arc_f, y_arc_f, marker='.', color='k', linestyle='')
            # cerchio = patches.Circle((x_arc_f, y_arc_f), r_arc_f, edgecolor='magenta', facecolor='none')
            # ax = plt.gca()
            # ax.add_patch(cerchio)

        else:
            x_arc_f = [x_v_arc_s[i] for i in range(n) if not exclude_flags[i]]
            y_arc_f = [y_v_arc_s[i] for i in range(n) if not exclude_flags[i]]
            r_arc_f = [x_inter_s[i] for i in range(n) if not exclude_flags[i]]
            print("Voids found:", len(x_v_arc))
            print("Voids after cleaning:", len(x_arc_f))

            # FINAL PLOT and SAVING
            # for b in range(len(x_arc_f)):
            # plt.plot(x_arc_f[b], y_arc_f[b], marker='.', color='k', linestyle='')
            # cerchio = patches.Circle((x_arc_f[b], y_arc_f[b]), r_arc_f[b], edgecolor='magenta', facecolor='none')
            # ax = plt.gca()
            # ax.add_patch(cerchio)
        
        header = 'R.A._center[arcmin]  Dec._center[arcmin]  R_v[arcmin]'
        np.savetxt(folderout + 'voids_radii.txt', np.c_[x_arc_f, y_arc_f, r_arc_f], 
                   fmt=' '.join(['%2.5f']*3), 
                   header=header, 
                   comments='')

    else:
        print("No voids found")

        # for b in range(len(x_v_arc_s)):
        # plt.plot(x_v_arc_s[b], y_v_arc_s[b], marker='.', color='k', linestyle='')
        # cerchio = patches.Circle((x_v_arc_s[b], y_v_arc_s[b]), x_inter_s[b], edgecolor='magenta', facecolor='none')
        # ax = plt.gca()
        # ax.add_patch(cerchio)#

    vmin = -threshold_value
    vmax = threshold_value    
        
    snr_map = -snr_map
    scaled_SNR = np.interp(snr_map, (vmin, vmax), (0, 1))
    cmap = cm.coolwarm
    summary_image = cmap(scaled_SNR)
    
    plt.rcParams['pgf.texsystem'] = 'pdflatex'
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1, 1, figsize=(16.5, 13.8))
    plt.axis('equal')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=25)
    plt.xlabel('$\mathrm{R. A. \;\;[arcmin]}$', fontsize=35)
    plt.ylabel('$\mathrm{Dec. \;\;[arcmin]}$', fontsize=35)
    plt.xlim(-ranges,ranges)
    plt.ylim(-ranges,ranges)
    plt.imshow(summary_image, origin='lower', extent=(-ranges, ranges, -ranges, ranges), aspect="equal", cmap=cmap,
               vmin=vmin, vmax=vmax)
    
    #fig.subplots_adjust(right=0.75)  # Regola questo valore per avvicinare/ allontanare la colorbar
    cb = plt.colorbar(ax=ax, pad=0.02) # fraction=0.046,
    cb.set_label("SNR", rotation=270, fontsize=35, labelpad=20)
    cb.ax.tick_params(labelsize=25)
    
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.2)  # "5%" è la larghezza della colorbar, pad è la distanza dal grafico

    # Aggiungi la colorbar
    #cb = plt.colorbar(mappable=ax.images[0], cax=cax)  # Usa ax.images[0] per collegare l'immagine alla colorbar
    #cb.set_label("SNR", rotation=270, fontsize=35, labelpad=20)
    #cb.ax.tick_params(labelsize=25)


    if exist:
        if n == 1:
            radius = r_arc_f
            circ = plt.Circle((x_arc_f, y_arc_f), radius, facecolor='none', alpha=1, edgecolor='gold', linewidth=4)
            ax.add_patch(circ)

            plt.plot(x_arc_f, y_arc_f, marker='x', markersize=4, color='white', markeredgecolor='gold',
                     markeredgewidth=1.2, zorder=10)
            circ_C = plt.Circle((x_arc_f, y_arc_f), radius, facecolor='none', fill=False, alpha=1, edgecolor='gold', linewidth=4)
            ax.add_patch(circ_C)
        else:
            for i in range(len(x_arc_f)):
                radius = r_arc_f[i]
                circ = plt.Circle((x_arc_f[i], y_arc_f[i]), radius, facecolor='none', alpha=1, edgecolor='gold', linewidth=4)
                ax.add_patch(circ)

                plt.plot(x_arc_f[i], y_arc_f[i], marker='x', markersize=4, color='white', markeredgecolor='gold',
                         markeredgewidth=1.2, zorder=10)
                circ_C = plt.Circle((x_arc_f[i], y_arc_f[i]), radius, facecolor='none', fill=False, alpha=1,
                                    edgecolor='k', linewidth=4)
                ax.add_patch(circ_C)
        # plt.text(-125, 140, 'SNR: 3.5 - 2', ha='center', va='center', bbox=dict(boxstyle='round', ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)), alpha=0.75)
    else:
        print("No voids found")
        
def calculate_max_min_r(data):
    c = int(data[0]) + 1
    max_r = c
    min_r = int(data[-1])
    return max_r, min_r

def size_function(radii, min_r, max_r, num_bins, log, bar, pltshow):
    
    valid_radii = [r for r in radii if min_r <= r <= max_r]
    
    fig, ax = plt.subplots(figsize=(70, 44))
    ax.set_xlabel("$R_\\mathrm{v} \;\;[\\mathrm{arcmin}]$", fontsize=100)
    ax.set_ylabel('Number of voids', fontsize=100)
    
    if log == False:
        hist, bins = np.histogram(valid_radii, bins=num_bins, range=(min_r, max_r))
        bin_centers = (bins[1:] + bins[:-1]) / 2
        #poisson_errors = 1 / np.sqrt(hist)
        poisson_errors = np.sqrt(hist)

        if bar:
            ax.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], align='center', alpha=0.7, color='blue', edgecolor='black', linewidth=2, label='Data')
            #ax.errorbar(bin_centers, hist, yerr=poisson_errors, fmt='o', color='k', markersize=20, label='Data')
            #ax.fill_between(bin_centers, hist - poisson_errors, hist + poisson_errors, color='gray', alpha=0.3)
        else:
            #ax.plot(bin_centers, hist, color='black', linewidth=2)
            ax.errorbar(bin_centers, hist, yerr=poisson_errors, fmt='o', color='k', markersize=20, label='Data')
            #ax.fill_between(bin_centers, hist - poisson_errors, hist + poisson_errors, color='gray', alpha=0.3, label='Data')

        ax.set_xlim([0, max_r])
        ax.set_xticks(np.arange(0, max_r + 1, step=1))
        ax.set_yticks(np.arange(0, max(hist) + 2, step=2))
        ax.tick_params(axis='both', labelsize=100)
        ax.grid(zorder=0, color='k', alpha=0.4, ls='--')
        ax.xaxis.set_tick_params(which='both', direction='in', length=1, pad=15)

    else:
        #logbins = np.logspace(np.log10(min_r), np.log10(max_r), num_bins + 1)
        #hist_l, bins_l = np.histogram(valid_radii, bins=logbins) 
        #bin_centers_l = np.sqrt(bins_l[1:] * bins_l[:-1])
        #poisson_errors_l = np.sqrt(hist_l)
        hist_l, bins_l = np.histogram(valid_radii, bins=num_bins, range=(min_r, max_r))
        #bin_centers = (bins[1:] + bins[:-1]) / 2
        #poisson_errors = np.sqrt(hist)
        #hist_l, bins_l = np.histogram(valid_radii, bins=linear_bins)
        bin_centers_l = (bins_l[1:] + bins_l[:-1]) / 2
        poisson_errors_l = np.sqrt(hist_l)
        #poisson_errors_l = 1 / np.sqrt(hist_l)

        if bar:
            ax.bar(bin_centers_l, hist_l, width=bin_centers_l[1] - bin_centers_l[0], align='center', alpha=0.7, color='blue', edgecolor='black', linewidth=2, label='Data')
            #ax.errorbar(bin_centers_l, hist_l, yerr=poisson_errors_l, fmt='o', color='k', markersize=20, label='Data')
            #ax.fill_between(bin_centers_l, hist_l - poisson_errors_l, hist_l + poisson_errors_l, color='gray', alpha=0.3)
        else:
            #ax.plot(bin_centers_l, hist_l, color='black', linewidth=2)
            ax.errorbar(bin_centers_l, hist_l, yerr=poisson_errors_l, fmt='o', color='k', markersize=20, label='Data')
            #ax.fill_between(bin_centers_l, hist_l - poisson_errors_l, hist_l + poisson_errors_l, color='gray', alpha=0.3, label='Data')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0, max_r])
        #ax.set_ylim([0, max(hist_l)+2])
        
        ax.grid(zorder=0, color='k', alpha=0.4, ls='--')
        #ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=100)
        #ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=100, pad=7)
        
        # y minor tick
        minor_tick_vals_y = [i for i in range(2, 10)] + [i*10 for i in range(2, 10)]
        minor_tick_labels_y = ["${}$".format(i) for i in range(2, 10)] + ["${}$".format(i*10) for i in range(2, 10)]
        ax.tick_params(axis='y', which='minor', labelsize=80)
        ax.yaxis.set_minor_locator(FixedLocator(minor_tick_vals_y))
        ax.yaxis.set_minor_formatter(FixedFormatter(minor_tick_labels_y))

        # y major tick
        major_tick_vals_y = [10**i for i in range(0, 2)]
        major_tick_labels_y = ["$10^{}$".format(i) for i in range(0, 2)]
        ax.set_yticks(major_tick_vals_y)
        ax.set_yticklabels(major_tick_labels_y, fontsize=100)

        # x minor tick
        #minor_tick_vals_x = [i for i in range(2, 10)] + [i*10 for i in range(2, 10)]
        #minor_tick_labels_x = ["${}$".format(i) for i in range(2, 10)] + ["${}$".format(i*10) for i in range(2, 10)]
        #ax.tick_params(axis='x', which='minor', labelsize=80)
        #ax.xaxis.set_minor_locator(FixedLocator(minor_tick_vals_x))
        #ax.xaxis.set_minor_formatter(FixedFormatter(minor_tick_labels_x))

        # x major tick
        major_tick_vals_x = [10**i for i in range(1, 2)]
        major_tick_labels_x = ["$10^{}$".format(i) for i in range(1, 2)]
        ax.set_xticks(major_tick_vals_x)
        ax.set_xticklabels(major_tick_labels_x, fontsize=100)
        
        #ax.yaxis.set_tick_params(which='both', direction='in', labelsize=100, pad=7)

        ax.yaxis.set_tick_params(which='minor', direction='in', width=2, length=15)
        ax.yaxis.set_tick_params(which='major', direction='in', width=4, length=30)
        #ax.xaxis.set_tick_params(which='minor', direction='in', width=2, length=15, pad=15)
        ax.xaxis.set_tick_params(which='major', direction='in', width=4, length=30, pad=20)
    
    ax.legend(fontsize=125, markerscale=2.5)  
    if pltshow:
        plt.show()
    else:
        plt.close()
    
    if log==False:
        return bin_centers, hist, poisson_errors
    else:
        return bin_centers_l, hist_l, poisson_errors_l
    
def map_vsf(folderout, num_bins, log=True, bar=False, pltshow=True):
    import Finder_functions as mystery
    
    radii = np.genfromtxt(folderout + 'voids_radii.txt', usecols=(2), unpack=True, skip_header=1)
    #print(radii)

    max_r, min_r = mystery.calculate_max_min_r(radii)

    bins, vsf, err_vsf = mystery.size_function(radii, min_r, max_r, num_bins, log, bar, pltshow)
    
    bins=np.array(bins)
    vsf=np.array(vsf)
    err_vsf=np.array(err_vsf)
    
    return max_r, min_r, bins, vsf, err_vsf

def plot_tot_vsf(pre_path, cosmoin, color, n, min_r, max_r, num_bins):
    import Finder_functions as mystery
            
# Creiamo un dizionario per le etichette delle cosmologie
    cosmo_labels_dict = {
        'LCDM': r'$\Lambda$CDM',
        'LCDM_0.15': r'$\Lambda$CDM$_{0.15 \, eV}$',
        'fR4': r'$f$R$4$',
        'fR5': r'$f$R$5$',
        'fR6': r'$f$R$6$'
    }
    if isinstance(cosmoin, str):
        cosmoin = [cosmoin]

    fig, ax = plt.subplots(figsize=(70, 44))

    for cos, col in zip(cosmoin, color):
        tot_vsf = []
        tot_errors_vsf = []
        bins = []

        subfolderss = [str(i).zfill(2) for i in range(n)]
        a = len(subfolderss)

        for i in subfolderss:
            folderout = pre_path + "/output_relative/" + cos + "/" + i + "/"
            radii = np.genfromtxt(folderout + 'voids_radii.txt', usecols=(2), unpack=True, skip_header=1)
            binss, vsf, err_vsf = mystery.size_function(radii, min_r, max_r, num_bins, log=True, bar=False, pltshow=False)

            # Aggiungi binss solo se non è già stato aggiunto
            if not bins:
                bins.extend(binss)

            tot_vsf.append(vsf)
            tot_errors_vsf.append(err_vsf)

        bins = np.array(bins)
        mean_vsf = np.mean(tot_vsf, axis=0)
        mean_vsf = np.array(mean_vsf)
        bar_vsf = np.std(tot_vsf, axis=0) / np.sqrt(a)
        bar_vsf = np.array(bar_vsf)

        # Plotta i dati per il cosmoin attuale
        ax.errorbar(bins, mean_vsf, yerr=bar_vsf, fmt='o', color=col, markersize=20, label=cosmo_labels_dict.get(cos, cos))
        ax.fill_between(bins, mean_vsf - bar_vsf, mean_vsf + bar_vsf, color=col, alpha=0.3)
        ax.grid(zorder=0, color=col, alpha=0.4, ls='--')

    # Personalizzazione del grafico
    ax.set_xlabel("$R_\\mathrm{v} \;\;[\\mathrm{arcmin}]$", fontsize=120)
    ax.set_ylabel('Number of voids', fontsize=120)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([2, 15])
    ax.set_ylim([0.09, max(mean_vsf)+7])

    #ax.errorbar(bins, mean_vsf, yerr=bar_vsf, fmt='o', color='k', markersize=20, label=f'VSF {cosmoin}')
    #ax.fill_between(bins, mean_vsf - bar_vsf, mean_vsf + bar_vsf, color='gray', alpha=0.3)
    #ax.grid(zorder=0, color='k', alpha=0.4, ls='--')

    # y minor tick
    minor_tick_vals_y = [i/10 for i in range(1, 10)]  
    minor_tick_labels_y = [''] * len(minor_tick_vals_y)
    minor_tick_vals_y += [i for i in range(1, 10)] + [i*10 for i in range(1, 10)]
    minor_tick_labels_y += [''] * len(minor_tick_vals_y)
    minor_tick_vals_y += [i*10 for i in range(1, 10)]
    minor_tick_labels_y += [''] * len(minor_tick_vals_y)

    ax.tick_params(axis='y', which='minor', labelsize=80)
    ax.yaxis.set_minor_locator(FixedLocator(minor_tick_vals_y))
    ax.yaxis.set_minor_formatter(FixedFormatter(minor_tick_labels_y))

    # y major tick
    major_tick_vals_y = [0.1, 1, 10]
    major_tick_labels_y = ["$10^{-1}$", "$10^{0}$", "$10^{1}$"]
    ax.set_yticks(major_tick_vals_y)
    ax.set_yticklabels(major_tick_labels_y, fontsize=110)

    # x minor tick
    #minor_tick_vals_x = [i for i in range(2, 10)] + [i*10 for i in range(2, 10)]
    #minor_tick_labels_x = ["${}$".format(i) for i in range(2, 10)] + ["${}$".format(i*10) for i in range(2, 10)]
    #ax.tick_params(axis='x', which='minor', labelsize=80)
    #ax.xaxis.set_minor_locator(FixedLocator(minor_tick_vals_x))
    #ax.xaxis.set_minor_formatter(FixedFormatter(minor_tick_labels_x))

    # x major tick
    major_tick_vals_x = [10**i for i in range(1, 2)]
    major_tick_labels_x = ["$10^{}$".format(i) for i in range(1, 2)]
    ax.set_xticks(major_tick_vals_x)
    ax.set_xticklabels(major_tick_labels_x, fontsize=110)

    ax.yaxis.set_tick_params(which='minor', direction='in', width=2, length=15)
    ax.yaxis.set_tick_params(which='major', direction='in', width=4, length=30)
    ax.xaxis.set_tick_params(which='minor', direction='in', width=2, length=15, pad=15)
    ax.xaxis.set_tick_params(which='major', direction='in', width=4, length=30, pad=20)
    
    ax.legend(fontsize=125, markerscale=2.5)
    
def bin_prof(x, y, nbins, ngal, sigmagal, rmin, rmax, cum=0):
    """
    Bin the profile, nbins set the number of bins,
    ngal number of background galaxies, for the error
    sigmagal is the intrinsic ellipticity distribution
    """
    #sì sfrutta la definizione di peso per fare la media dei pixel contenuti in ogni shell ed estrarci la convergenza

    x = np.array(x) #distanze dei pixel dal centro del vuoto (r)
    y = np.array(y) #valore della convergenza in ogni pixel (k(r))
    r = x 
    kappa = y
    y2 = y*y
    if nbins<0:
        xx = np.logspace(np.log10(rmin), np.log10(rmax), -nbins) 
    else:
        xx = np.linspace(rmin, rmax, nbins) #binning (da 0 a 10 Rv)

    a, b0 = np.histogram(x, bins=xx) #a:conteggio dei pixel che cadono in ogni bin (hist, bin_edges) .
    c, b = np.histogram(x, bins=xx, weights=y) #c:conteggio dei valori di convergenza nei pixel che cadono in ogni bin: c (somma dei valori di convergenza)
    e, d = np.histogram(x, bins=xx, weights=y2) #e (somma dei quadrati dei valori di convergenza) 
    #b0=b=d bordo dei bin per ogni vuoto 
    #print(a)
    #print(c)
    #print(e)
    x = ((b[1:] + b[:-1])/2.0) #ridefinisce il centro dei bin
    ss = sigmagal/np.sqrt(3600*np.pi*ngal*(b0[1:]**2-b0[:-1]**2))    #shape noise/3600
    s = sigmagal/np.sqrt(np.pi*ngal*(b0[1:]**2-b0[:-1]**2))    #shape noise (salvo questo)
    y = (c / a) #valore di convergenza media in ogni shell (bin) dividendo la somma dei valori di convergenza tra tutti i pixel della shell (c) per il numero di pixel (a) della shell [ sum(k_pixel) / n_pixel ]
    y_cum = y - y 
    
    if cum==1:
        for i in range(0,len(x)):
            yin = kappa[r<=x[i]] #Estrae i valori di convergenza kappa per i pixel che si trovano a una distanza r minore o uguale a x[i]
            y_cum[i] = np.nanmean(yin) #valore di convergenza media entro (contenuta) un certo raggio di una shell
    y = np.nan_to_num(y) 
    e = np.sqrt(e/a - y*y) #calcola la deviazione standard della convergenza nei bin come la radice quadrata della differenza tra la media dei quadrati delle convergenze (divisa per il numero di pixel a) e il quadrato della convergenza media y in ogni shell
    error_y = e / np.sqrt(a)  # Errore standard della media per y (errore sul valore di convergenza in una shell)
    error_y_cum = np.sqrt(np.cumsum(error_y**2))  # Errore sulla media cumulativa calcolato come la propagazione dell'errore standard su tutte le osservazioni fino al raggio della shell considerata
    error_y = np.nan_to_num(error_y)
    error_y_cum = np.nan_to_num(error_y_cum)
    e = np.nan_to_num(e)
    s = np.nan_to_num(s) #shape noise corretto
    ss = np.nan_to_num(ss)
    
    if cum==1:
        return x, y, error_y, y_cum, error_y_cum, e, a, s, ss #sono liste di 64 valori, uno per ogni shell
    else:
        return x, y, error_y, e, a, s

def bin_prof_log(x, y, nbins_log, ngal, sigmagal, rmin, rmax, cum=0): 
    x = np.array(x)
    y = np.array(y)
    r = x 
    kappa = y
    y2 = y*y
    
    if nbins_log<0:
        xx = np.logspace(np.log10(rmin), np.log10(rmax), -nbins_log)
    
    else:
        xx = np.logspace(np.log10(rmin), np.log10(rmax), nbins_log) #log binning
    a, b0 = np.histogram(x, bins=xx)
    c, b = np.histogram(x, bins=xx, weights=y)
    e, d = np.histogram(x, bins=xx, weights=y2)
    x = ((b[1:] + b[:-1])/2.0)
    s = sigmagal/np.sqrt(3600*np.pi*ngal*(b0[1:]**2-b0[:-1]**2))
    y = (c / a)
    y_cum = y - y
    
    if cum==1:
        for i in range(0,len(x)):
            yin = kappa[r<=x[i]]
            y_cum[i] = np.nanmean(yin)
    y = np.nan_to_num(y)
    e = np.sqrt(e/a - y*y) 
    e = np.nan_to_num(e)
    s = np.nan_to_num(s)
    
    if cum==1:
        return x, y, s, e, a, y_cum
    
    else:
        return x, y, s, e, a

def calculate_param_intervals(samples, percentiles, num_params):
    param_intervals = np.percentile(samples, percentiles, axis=0)

    param_fitted = param_intervals[1]

    param_errors = 0.5 * (param_intervals[2] - param_intervals[0])

    if len(param_fitted) != num_params:
        raise ValueError("The number of fitted parameters is not consistent with the initial number of parameters.")

    return param_fitted, param_errors

def shells_profile(folderout, i_fil, smooth_filter_set, folder_names, kappa_final, equibins, fov_arcmin, npixel, n_gal_set, ell_err, nbins, fradius, graph=True):
    import Finder_functions as mystery
    
    x_v_arc, y_v_arc, r_v_arc = np.genfromtxt(folderout + 'voids_radii.txt', unpack=True, skip_header=1)

    if x_v_arc.size == 1:
        points = [(r_v_arc, x_v_arc, y_v_arc)]
    else:
        points = zip(r_v_arc,x_v_arc,y_v_arc)
    sorted_points = sorted(points)

    r_v_arc  = np.array([point[0] for point in sorted_points])
    x_v_arc  = np.array([point[1] for point in sorted_points])
    y_v_arc  = np.array([point[2] for point in sorted_points])

    npeaks=len(x_v_arc)
    #print('n total voids = ', npeaks)

    map = np.copy(kappa_final)*0.0 - 100
    distances = map - map 
    x = np.linspace(0, npixel-1, npixel)
    x = np.int32(x) 
    xx_, yy_ = np.meshgrid(x, x)

    if graph:
        plt.figure(figsize=(18, 14)) 
        plt.xlim(0,fradius*0.5)
        plt.ylim(-13, 6)
        plt.xticks(np.arange(0, (fradius*0.5)+1, step=1))     
        plt.xlabel('$r_p/R_v$', fontsize=45)
        plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=45)
        plt.yticks(fontsize=35)
        plt.xticks(fontsize=35)
        #plt.gca().yaxis.set_tick_params(which='both', direction='in', labelsize=25, length=4, width=1)
        #plt.gca().xaxis.set_tick_params(which='both', direction='in', labelsize=25, length=4, width=1)
        #plt.legend(fontsize=25)

    for i in range(len(equibins)-1):
        min_radius = equibins[i]
        max_radius = equibins[i+1]

        for j in range(0,npeaks):
            radius = r_v_arc[j]
            if min_radius <= radius < max_radius:
                d = fradius*radius
                center = x_v_arc[j], y_v_arc[j]
                print('void', j+1,':', center,'radius: ', radius,'arcmin')
                x_ = np.int32((x_v_arc[j])/fov_arcmin*npixel + npixel*0.5) 
                y_ = np.int32((y_v_arc[j])/fov_arcmin*npixel + npixel*0.5)
                distances = np.sqrt((xx_-x_)**2 + (yy_-y_)**2) #calcola le distanze tra i punti della griglia xx_ e le coordinate dei centri dei vuoti in pixel 
                dx1 = np.int32((x_v_arc[j]-0.5*d)/fov_arcmin*npixel + npixel*0.5)
                dx2 = np.int32((x_v_arc[j]+0.5*d)/fov_arcmin*npixel + npixel*0.5)  
                dy1 = np.int32((y_v_arc[j]-0.5*d)/fov_arcmin*npixel + npixel*0.5)  
                dy2 = np.int32((y_v_arc[j]+0.5*d)/fov_arcmin*npixel + npixel*0.5)

                if dx2>2047:
                    dx2 = 2047
                if dy2>2047:
                    dy2 = 2047
                if dx1<0:
                    dx1=0
                if dy1<0:
                    dy1=0
                dx = dx2-dx1 
                dy = dy2-dy1

                map_ = np.zeros([dy,dx]) 
                dmap_ = np.zeros([dy,dx])

                map[dy1:dy2,dx1:dx2] = kappa_final[dy1:dy2,dx1:dx2]
                map_[0:dy,0:dx] = kappa_final[dy1:dy2,dx1:dx2]
                dmap_[0:dy,0:dx] = distances[dy1:dy2,dx1:dx2]*(fov_arcmin/npixel)

                map_ = map_.flatten()
                dmap_ = dmap_.flatten()

                r_shell, k_shell, k_errors, k_cum_shell, k_cum_errors, std_k_shell, N_pixel_shell, GSN_shell, GSN_3600_shell = mystery.bin_prof(dmap_, map_, nbins+1, n_gal_set, ell_err, 0, 0.5*d, 1)

                r_out = r_shell/radius
                #print(r_out)

                g_t = (k_cum_shell - k_shell) #reduced tangential shear
                gt_errors = np.sqrt(k_cum_errors**2 + k_errors**2) #errors propagation on g_t
                #print(g_t)
                #print(GSN_shell)
                #print(GSN_3600_shell)
                
                GSN_shell_rel = GSN_shell/g_t
                #print(GSN_shell_rel*100)

                filename = folderout + folder_names[i] + '/' + 'void_' + str(j) + '.txt'
                
                header = 'r_shell/R_v  K_ENCLOSED_shell  K_shell  ERROR_K_ENCLOSED  ERROR_K_shell  STD_K_shell  GSN_shell  ERROR_gt_Kmeasure_shell'

                with open(filename, "w") as f:
                    f.write(f'R_v = {r_v_arc[j]:.7f} arcmin\n')

                    np.savetxt(f, np.c_[r_out, k_cum_shell, k_shell, k_cum_errors, k_errors, std_k_shell, GSN_shell, gt_errors], 
                               header=header, 
                               fmt=' '.join(['%2.7f']*8),
                               comments='')
                if graph:
                    #plt.plot(r_out[N_pixel_shell>8], k_shell[N_pixel_shell>8])
                    ##plt.errorbar(r_out, g_t*1e3, GSN_shell*1e3, color='gray', linewidth=0.5, alpha=0.5)
                    ##plt.errorbar(r_out, g_t*1e3, gt_errors*1e3, color='gray', linewidth=0.5, alpha=0.5)
                    plt.plot(r_out, g_t*1e3, color='gray', linewidth=0.5, alpha=0.5)
                    #if j==0:
                        #plt.errorbar(r_out, g_t*1e3, gt_errors*1e3, color='dodgerblue', linewidth=2)
                    #if j==41:
                        #plt.errorbar(r_out, g_t*1e3, gt_errors*1e3, color='green', linewidth=2)   
                    #if j==81:
                        #plt.errorbar(r_out, g_t*1e3, gt_errors*1e3, color='red', linewidth=2)

    return r_out, nbins, fradius

def process_shear_map(folderout, l_c, dirs, color, r_out, nbins, fradius, plot):
    import Finder_functions as mystery
    bin_edges = [1, 3.79, 4.85, 15]
    path_rp = folderout + dirs[0] + "/"
    radii = list(np.genfromtxt(path_rp + 'void_0.txt', usecols=(0), unpack=True, skip_header=2))        
    rp = np.array(radii)
    
    if plot == True:    
        plt.figure(figsize=(14, 14))
        plt.xlim(0, 10)
        plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1)) 
        plt.ylim(-8, 0.5)
        plt.xlabel('$r_p/R_v$', fontsize=35)
        plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=35)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)

    shear__map = []
    shear_errors_map = []
    GSN_errors_map = []
    lc_counter = 0
    
    for ii, d in enumerate(dirs):
        dir_counter = 0
        shear__dir = []
        min_radius = bin_edges[ii]
        max_radius = bin_edges[ii+1]
        labelbin = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'
        
        folderon = folderout + d + "/"
        file_list = [file for file in glob.glob(folderon + '*') if not file.endswith('_log.txt')]
        file_list = sorted(file_list, key=lambda x: int(re.search(r'void_(\d+)\.txt', x).group(1)))

        for filename in file_list:
            _, kappa, kappa_shell, GSN_shell, shear_error_shell = np.genfromtxt(filename, usecols=(0, 1, 2, 6, 7), unpack=True, skip_header=2)

            shear = kappa - kappa_shell
            var = True

            if var:
                lc_counter += 1
                dir_counter += 1
                shear__map.append(shear)
                shear__dir.append(shear)
                GSN_errors_map.append(GSN_shell) # # Add errors from galaxy shape noise
                shear_errors_map.append(shear_error_shell)  # Add errors from measure
        
        shear_dir = np.array(shear__dir)
        dir_map_mean_shear = np.mean(shear_dir, axis=0)
        dir_map_std_shear = np.std(shear_dir, axis=0) #/ np.sqrt(dir_counter)
        dir_shear_error_cov, dir_shear_error_jack, dir_shear_error_boot = mystery.calculate_stat_error(shear_dir, nbins, len(file_list))
        dir_shear_error_cov = mystery.round_up_decimal(dir_shear_error_cov, 7)
        dir_shear_error_jack = mystery.round_up_decimal(dir_shear_error_jack, 7)
        dir_shear_error_boot = mystery.round_up_decimal(dir_shear_error_boot, 7)
        dir_map_mean_shear = mystery.round_up_decimal(dir_map_mean_shear, 7)
        dir_map_std_shear = mystery.round_up_decimal(dir_map_std_shear, 7)
        if plot == True:    
            plt.errorbar(rp, dir_map_mean_shear*1e3, dir_shear_error_jack*1e3, label=f'{labelbin}', linestyle='-', color=color[ii])


    shear_map = np.array(shear__map)
    np.savetxt(folderout + 'map_shear_profiles', shear_map, fmt='%2.7f')
    
    GSN_errors_map = np.array(GSN_errors_map)
    shear_errors_map = np.array(shear_errors_map)

    map_mean_shear = np.mean(shear_map, axis=0)
    map_std_shear = np.std(shear_map, axis=0) / np.sqrt(lc_counter)
    
    mean_shear_error = np.sqrt(np.sum(shear_errors_map**2, axis=0)) / lc_counter
    mean_GSN_shear_error = np.sqrt(np.sum(GSN_errors_map**2, axis=0)) / lc_counter

    weights = 1 / shear_errors_map**2
    weighted_mean_shear = np.sum(shear_map * weights, axis=0) / np.sum(weights, axis=0)

    weights_GSN = 1 / GSN_errors_map**2
    weighted_GSN_mean_shear = np.sum(shear_map * weights_GSN, axis=0) / np.sum(weights_GSN, axis=0)

    weighted_mean_shear_error = 1 / np.sqrt(np.sum(weights, axis=0))
    weighted_GSN_mean_shear_error = 1 / np.sqrt(np.sum(weights_GSN, axis=0))

    map_shear_error_cov, map_shear_error_jack, map_shear_error_boot = mystery.calculate_stat_error(shear_map, nbins, 100)
    map_shear_error_cov = mystery.round_up_decimal(map_shear_error_cov, 7)
    map_shear_error_jack = mystery.round_up_decimal(map_shear_error_jack, 7)
    map_shear_error_boot = mystery.round_up_decimal(map_shear_error_boot, 7)

    map_mean_shear = mystery.round_up_decimal(map_mean_shear, 7)
    map_std_shear = mystery.round_up_decimal(map_std_shear, 7)
    
    mean_shear_error = mystery.round_up_decimal(mean_shear_error, 7)
    mean_GSN_shear_error = mystery.round_up_decimal(mean_GSN_shear_error, 7)
    
    weighted_mean_shear = mystery.round_up_decimal(weighted_mean_shear, 7)
    weighted_mean_shear_error = mystery.round_up_decimal(weighted_mean_shear_error, 7)
    
    weighted_GSN_mean_shear = mystery.round_up_decimal(weighted_GSN_mean_shear, 7)
    weighted_GSN_mean_shear_error = mystery.round_up_decimal(weighted_GSN_mean_shear_error, 7)
    
    # Save for map l_c
    map_filename = folderout + 'map_' + l_c + '_mean_shear.dat'
    header = 'r/R_v  gt_MEAN  ERROR_COVMAT  ERROR_JACKKNIFE  ERROR_BOOTSTRAP  ESM_Kmeasure  ESM_GSN  gt_WEIGHTED_Kmeasure  ERROR_gt_WEIGHTED_Kmeasure  gt_WEIGHTED_GSN  ERROR_gt_WEIGHTED_GSN'

    with open(map_filename, 'w') as file:
        np.savetxt(file, 
                   list(zip(rp, map_mean_shear, map_shear_error_cov, map_shear_error_jack, map_shear_error_boot, 
                            mean_shear_error, mean_GSN_shear_error, weighted_mean_shear, weighted_mean_shear_error, 
                            weighted_GSN_mean_shear, weighted_GSN_mean_shear_error)), 
                   fmt=' '.join(['%2.7f']*11), 
                   header=header, 
                   comments='')
    
    if plot == True:    
        plt.errorbar(rp, map_mean_shear*1e3, map_shear_error_cov*1e3, label='All voids of the map', linestyle='--', color='k')
        plt.legend(fontsize=40)
        
    _, _, _, _ = mystery.calculate_covariance_matrix(shear_map, plot)

def load_n_profiles(pre_path, cosmoc, n):
    folderin = os.path.join(pre_path, 'output_relative', cosmoc)
    
    subfolderss = [str(l_c).zfill(2) for l_c in range(n)]  # Sottocartelle da "00" a "n-1"
    profiles_list = []  # Lista per accumulare tutte le righe di tutti i file
    
    for l_c in subfolderss:
        profile_path = os.path.join(folderin, l_c, "map_shear_profiles")
        
        if not os.path.exists(profile_path):
            print(f"File non trovato: {profile_path}")
            continue

        if os.path.isfile(profile_path):
            #print(f"Leggendo: {profile_path}")
            try:
                data = np.genfromtxt(profile_path, dtype=float)  # Legge il file
                
                if data.ndim == 1:  # Caso in cui il file ha solo 1 riga (reshape per coerenza)
                    data = data.reshape(1, -1)

                if data.shape[1] == 64:  # Controlla che ogni riga abbia esattamente 64 valori
                    profiles_list.append(data)
                else:
                    print(f"ERRORE: {profile_path} ha {data.shape[1]} colonne invece di 64!")
            
            except Exception as e:
                print(f"ERRORE nella lettura di {profile_path}: {e}")
    
    if profiles_list:
        profiles__ = np.vstack(profiles_list)  # Concatena tutte le righe in un unico array
        print(f"Caricati {profiles__.shape[0]} righe con {profiles__.shape[1]} colonne ciascuna.")
    else:
        print("Nessun dato valido trovato.")
        profiles__ = np.array([])

    return profiles__
        
def create_custom_diverging_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    # Custom diverging colormap from blue to white to deep red
    colors = [
        (0.0, "lightblue"),         # -0.2e-5
        (0.2, "white"),
        (0.4, "lightcoral"),      # 0.0 (centro visivo)
        (0.6, "indianred"),
        (0.8, "firebrick"),
        (1.0, "darkred")       # 1.0e-5
    ]
    return LinearSegmentedColormap.from_list("custom_div_cmap", colors)
        
def calculate_covariance_matrix(profiles, plot=False):
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MultipleLocator

    covariance_matrix = np.cov(profiles, rowvar=False)
    # Extract the diagonal values
    diagonal_values = np.diag(covariance_matrix)
    # Compute the square root of the diagonal values
    sqrt_diagonal_values = np.sqrt(diagonal_values)
    corr_mat = np.corrcoef(profiles, rowvar=False)
    
    # Compute the square root of the diagonal values
    # Calcola la radice quadrata dei valori diagonali, con un controllo sui valori negativi o zero
    #sqrt_diagonal_values = np.where(diagonal_values >= 0, np.sqrt(diagonal_values), 0)

    # Calcola la matrice di correlazione
    #corr_mat = np.corrcoef(profiles, rowvar=False)

    # Gestione eventuali valori non validi (NaN o inf) nella matrice di covarianza
    #covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    #corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
    
    if plot == True:
        
        vmin, vmax = -0.2e-5, 0.9e-5
        # Define custom normalization so that 0.0 falls exactly in the middle color (white)
        norm = Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(12, 12))
        cmap = create_custom_diverging_colormap()
        im = ax.imshow(covariance_matrix, origin='upper', cmap=cmap, norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)

        ticks = np.arange(-0.2e-5, 1e-5, 0.2e-5)
        cbar = plt.colorbar(im, cax=cax, ticks=ticks)
        cbar.ax.tick_params(labelsize=25)
        cbar.ax.yaxis.offsetText.set_fontsize(20)
        cbar.ax.yaxis.offsetText.set_x(1.65)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Covariance', size=30, rotation=270)

        ax.tick_params(which='both', direction='in', labelsize=25)
        #ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=25)
        #ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=25)
        plt.tight_layout()
        plt.savefig("/home/leonardo/Desktop/pdfs2/cov_mat_all_LCDM.pdf", format='pdf', bbox_inches='tight')
        plt.show()

        
        #Plot Correlation factor matrix
        fig, ax = plt.subplots(1,1, figsize=(12,12))

        im = plt.imshow(corr_mat, origin='upper')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)

        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=15) 

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Correlation factor', size=22, rotation=270)

        ax.get_xaxis().set_tick_params(which='both', direction='in', labelsize=15)  
        ax.get_yaxis().set_tick_params(which='both', direction='in', labelsize=15)

    return covariance_matrix, diagonal_values, sqrt_diagonal_values, corr_mat

def ensure_positive_definite(matrix, eps_tolerance=1e-10, enforce_eps=1e-8, verbose=True):
    """
    Rende positiva definita una matrice simmetrica con boost dinamico in caso di autovalori negativi residui.

    Parametri:
        matrix (np.ndarray): matrice simmetrica NxN
        eps_tolerance (float): soglia per considerare un autovalore nullo/negativo numericamente
        enforce_eps (float): soglia minima per forzare autovalori molto piccoli o negativi
        verbose (bool): se True, stampa diagnostica

    Ritorna:
        corrected_matrix (np.ndarray): matrice positiva definita
        float: autovalore minimo finale
    """
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    min_eig = np.min(eigvals)

    if min_eig > -eps_tolerance:
        if verbose:
            print("La matrice è già definita positiva (entro tolleranza numerica).")
        return matrix, min_eig

    if verbose:
        print(f"Autovalore minimo negativo: {min_eig:.2e}. Correzione necessaria.")

    # Correzione iniziale autovalori
    corrected_eigvals = np.clip(eigvals, enforce_eps, None)
    corrected_matrix = eigvecs @ np.diag(corrected_eigvals) @ eigvecs.T

    final_min = np.min(np.linalg.eigvalsh(corrected_matrix))
    if final_min < -eps_tolerance:
        if verbose:
            print(f"Dopo correzione, autovalore minimo ancora negativo: {final_min:.2e}. Applico boost dinamico.")
        # Boost più forte e adattivo
        boost = abs(final_min) * 10 + enforce_eps
        corrected_matrix += boost * np.eye(matrix.shape[0])
        final_min = np.min(np.linalg.eigvalsh(corrected_matrix))

        if final_min > 0:
            if verbose:
                print(f"Boost applicato: +{boost:.2e} → nuovo min λ = {final_min:.2e}")
        else:
            if verbose:
                print(f"Boost applicato: +{boost:.2e}, ma la matrice NON è ancora definita positiva. Min λ = {final_min:.2e}")

    return corrected_matrix


def calculate_stat_error(profiles__, nbins, n, plot_cov=False):
    import Finder_functions as mystery
    
    # Converte la lista in un array per facilitare le operazioni
    profiles__ = np.array(profiles__)
    #print(profiles__)
    
    # Calcola il numero totale di profili
    num_profiles = profiles__.shape[0]

    covariance_matrix, diagonal_values, sqrt_diagonal_values, corr_mat = mystery.calculate_covariance_matrix(profiles__)
    # --- Pulizia iniziale ---
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Filtro sulla correlazione ---
    correlation_threshold = 0.2
    for i in range(nbins):
        for j in range(nbins):
            if i != j and abs(corr_mat[i, j]) < correlation_threshold:
                covariance_matrix[i, j] = 0.0

    # --- Regolarizzazione per stabilità numerica ---
    reg_epsilon = 1e-5 * np.mean(np.diag(covariance_matrix))
    covariance_matrix += reg_epsilon * np.eye(nbins)

    # --- Forza positività definita sulla matrice di covarianza ---
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    eigvals_clipped = np.clip(eigvals, a_min=1e-10, a_max=None)
    covariance_matrix = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # --- Calcolo errore finale (opzionale) ---
    final_cov_error = sqrt_diagonal_values / np.sqrt(num_profiles)

    # --- Inversione della matrice ---
    inverse_cov_mat = np.linalg.inv(covariance_matrix)

    # --- Correzione con fattore di Hartlap ---
    hartlap_factor = (num_profiles - nbins - 2) / (num_profiles - 1)
    if hartlap_factor > 0:
        inverse_cov_mat_hartlap = inverse_cov_mat * hartlap_factor
    else:
        print("Hartlap factor negativo o nullo. Usiamo la matrice non corretta.")
        inverse_cov_mat_hartlap = inverse_cov_mat.copy()

    # --- Pulizia post-inversione ---
    inverse_cov_mat_hartlap = np.nan_to_num(inverse_cov_mat_hartlap, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Filtro outlier numerici (> 3σ off-diagonali) ---
    off_diag_values = inverse_cov_mat_hartlap[~np.eye(nbins, dtype=bool)]
    std_off = np.std(off_diag_values)

    for i in range(nbins):
        for j in range(nbins):
            if i != j and abs(inverse_cov_mat_hartlap[i, j]) > 3 * std_off:
                inverse_cov_mat_hartlap[i, j] = 0.0

    # --- Forza definita positività della matrice finale ---
    inverse_cov_mat_hartlap = mystery.ensure_positive_definite(inverse_cov_mat_hartlap, eps_tolerance=1e-10, enforce_eps=1e-8, verbose=True)

    # (Facoltativo) Controllo finale
    #eigvals_final = np.linalg.eigvalsh(inverse_cov_mat_hartlap)
    #print("Minimo autovalore finale:", np.min(eigvals_final))
    
    #######################################################################à###############################
    
    #Prendi la diagonale dell'inversa corretta
    diag_inverse_cov_mat_hartlap = np.diag(inverse_cov_mat_hartlap)
    
    # Reinverti la matrice corretta
    cov_mat_hartlap = np.linalg.inv(inverse_cov_mat_hartlap)
    
    #Prendi la diagonale della matrice corretta e reinvertita
    diag_cov_mat_hartlap = np.diag(cov_mat_hartlap)
    
    ## 1. Calcola errori attraverso cov mat con correzione di hartlap
    final_cov_error_hartlap = np.sqrt(diag_cov_mat_hartlap)/np.sqrt(num_profiles)
        
    
    # Inversione della matrice di covarianza
    #try:
    #    inverse_cov_mat = np.linalg.inv(covariance_matrix)
    #except np.linalg.LinAlgError:
    #    print("Errore: Matrice di covarianza non invertibile.")
    #    inverse_cov_mat = np.zeros_like(covariance_matrix)

    # Correzione di Hartlap
    #try:
    #    inverse_cov_mat_hartlap = inverse_cov_mat * ((num_profiles - nbins - 2) / (num_profiles - 1))
    #    cov_mat_hartlap = np.linalg.inv(inverse_cov_mat_hartlap)
    #    final_cov_error_hartlap = np.sqrt(np.diag(cov_mat_hartlap)) / np.sqrt(num_profiles) ###
    #except np.linalg.LinAlgError:
    #    print("Errore: Matrice di covarianza con correzione di Hartlap non invertibile.")
    #    final_cov_error_hartlap = np.full(nbins, np.nan)
    
    
    ## 2. Calcola errori attraverso jackknife sottraendo un profilo per volta
    jackknife_means = []
    total_mean = np.mean(profiles__, axis=0)  # Profilo medio globale
    
    # Escludi ogni singolo profilo uno volta alla volta
    for i in range(num_profiles):
        profiles_saved = np.delete(profiles__, i, axis=0)  # Escludi il profilo corrente
        jackknife_mean = np.mean(profiles_saved, axis=0)  # Calcola la media dei profili rimanenti
        jackknife_means.append(jackknife_mean)

    # Converti in array e calcola la varianza jackknife
    jackknife_means = np.array(jackknife_means)
    jackknife_variance = ((num_profiles - 1) / num_profiles) * np.sum((jackknife_means - total_mean) ** 2, axis=0)
    jackknife_error = np.sqrt(jackknife_variance)  ###

    ## 3. Calcola errori attraverso bootstrap con n campioni
    bootstrap_means = []

    # Applica il Bootstrap: genera campioni con sostituzione
    for i in range(n):
        # Crea un campione bootstrap con sostituzione
        bootstrap_sample = profiles__[np.random.randint(0, num_profiles, size=num_profiles)]

        # Calcola la media sui profili del campione bootstrap
        bootstrap_mean = np.mean(bootstrap_sample, axis=0)
        bootstrap_means.append(bootstrap_mean)

    # Converti in array per facilità di calcolo
    bootstrap_means = np.array(bootstrap_means)

    # Calcola la varianza sulle medie bootstrap
    bootstrap_variance = np.var(bootstrap_means, axis=0, ddof=1)

    # Calcola l'errore (deviazione standard) dal bootstrap
    bootstrap_error = np.sqrt(bootstrap_variance) ###
    
    return final_cov_error, jackknife_error, bootstrap_error, final_cov_error_hartlap, inverse_cov_mat_hartlap

def round_up_decimal(arr, n):
    factor = 10**n
    return np.sign(arr) * (np.ceil(np.abs(arr) * factor) / factor)

def process_shear_1binsize(pre_path, cosmoin, dirs, n, nbins):
    import Finder_functions as mystery
    rp = None

    for i in range(len(cosmoin)):
        ff_counter = 0
        print(f'{cosmoin[i]}')
        shear__ = []
        shear_errors__ = []
        GSN_errors__ = []
        subfolderss = [str(l_c).zfill(2) for l_c in range(n)]
        
        if rp is None and i == 0:
            path_rp = pre_path + '/output_relative/' + cosmoin[i] + "/" + subfolderss[0] + "/" + dirs[0] + "/"
            radii = list(np.genfromtxt(path_rp + 'void_0.txt', usecols=(0), unpack=True, skip_header=2))
            rp = np.array(radii)
            
        for l_c in subfolderss:
            lc_counter = 0
            shear__map = []
            shear_errors_map = []
            GSN_errors_map = []
            folderin = pre_path + '/output_relative/' + cosmoin[i] + "/" + l_c + "/"

            for d in dirs:
                folderon = folderin + d + "/"
                file_list = [file for file in glob.glob(folderon + '*') if not file.endswith('_log.txt')]
                file_list = sorted(file_list, key=lambda x: int(re.search(r'void_(\d+)\.txt', x).group(1)))
            
                for filename in file_list:
                    _, kappa, kappa_shell, GSN_shell, shear_error_shell = np.genfromtxt(filename, usecols=(0, 1, 2, 6, 7), unpack=True, skip_header=2)
                    
                    shear = kappa - kappa_shell
                    var = True
                    
                    if var:
                        ff_counter += 1
                        shear__map.append(shear)
                        GSN_errors_map.append(GSN_shell) # # Add errors from galaxy shape noise
                        shear_errors_map.append(shear_error_shell)  # Add errors from measure
            
            shear_map = np.array(shear__map)
            GSN_errors_map = np.array(GSN_errors_map)
            shear_errors_map = np.array(shear_errors_map)
        
            if shear_map.ndim == 1:
                shear_map = shear_map[np.newaxis, :]
                GSN_errors_map = GSN_errors_map[np.newaxis, :]
                shear_errors_map = shear_errors_map[np.newaxis, :]             

            shear__.append(shear_map)
            GSN_errors__.append(GSN_errors_map)
            shear_errors__.append(shear_errors_map)
            
        shear__ = np.vstack(shear__)
        GSN_errors__ = np.vstack(GSN_errors__)
        shear_errors__ = np.vstack(shear_errors__)

        if len(shear__) > 0:
            mean_shear = np.mean(shear__, axis=0)
            std_shear = np.std(shear__, axis=0) / np.sqrt(shear__.shape[0])
            
            # Calculate propagation errors on mean shear
            mean_shear_error = np.sqrt(np.sum(shear_errors__**2, axis=0)) / shear__.shape[0]
            mean_GSN_shear_error = np.sqrt(np.sum(GSN_errors__**2, axis=0)) / shear__.shape[0]
            
            # Calculated weighted mean
            weights = 1 / shear_errors__**2
            weighted_mean_shear = np.sum(shear__ * weights, axis=0) / np.sum(weights, axis=0)
            
            weights_GSN = 1 / GSN_errors__**2
            weighted_GSN_mean_shear = np.sum(shear__ * weights_GSN, axis=0) / np.sum(weights_GSN, axis=0)

            # Calculate error propagation on weighted mean
            weighted_mean_shear_error = 1 / np.sqrt(np.sum(weights, axis=0))
            
            weighted_GSN_mean_shear_error = 1 / np.sqrt(np.sum(weights_GSN, axis=0))

            # Calculate statistical errors on mean shear of cosmology
            if i==0:
                
                shear_error_cov, shear_error_jack, shear_error_boot, shear_error_cov_H, inverse_cov_H = mystery.calculate_stat_error(shear__, nbins, 10000, True)
            else:
                shear_error_cov, shear_error_jack, shear_error_boot, shear_error_cov_H, inverse_cov_H = mystery.calculate_stat_error(shear__, nbins, 10000, False)
                
            shear_error_cov = mystery.round_up_decimal(shear_error_cov, 7)
            shear_error_cov_H = mystery.round_up_decimal(shear_error_cov_H, 7)
            shear_error_jack = mystery.round_up_decimal(shear_error_jack, 7)
            shear_error_boot = mystery.round_up_decimal(shear_error_boot, 7)
            
            
            mean_shear = mystery.round_up_decimal(mean_shear, 7) #mean shear
            std_shear = mystery.round_up_decimal(std_shear, 7) #standard deviation shear
            
            mean_shear_error = mystery.round_up_decimal(mean_shear_error, 7) #ESM from propagation of measure K_cum - K
            mean_GSN_shear_error = mystery.round_up_decimal(mean_GSN_shear_error, 7) #ESM from propagation of GSN
            
            weighted_mean_shear = mystery.round_up_decimal(weighted_mean_shear, 7) #WM with weights from measure K_cum - K
            weighted_mean_shear_error = mystery.round_up_decimal(weighted_mean_shear_error, 7) #errors on WM from measure K_cum - K propagation
            
            weighted_GSN_mean_shear = mystery.round_up_decimal(weighted_GSN_mean_shear, 7) #WM with weights from GSN
            weighted_GSN_mean_shear_error = mystery.round_up_decimal(weighted_GSN_mean_shear_error, 7) #errors on WM from GSN
            
            print(ff_counter)
                    
            # Save files for cosmoin[i]
            output_filename = pre_path + '/output_relative/' + f'{cosmoin[i]}_mean_shear.dat'
            header = 'r/R_v  gt_MEAN  ERROR_COVMAT  ERROR_JACKKNIFE  ERROR_BOOTSTRAP  ESM_Kmeasure  ESM_GSN  gt_WEIGHTED_Kmeasure  ERROR_gt_WEIGHTED_Kmeasure  gt_WEIGHTED_GSN  ERROR_gt_WEIGHTED_GSN'

            with open(output_filename, 'w') as file:
                np.savetxt(file, 
                           list(zip(rp, mean_shear, shear_error_cov, shear_error_jack, shear_error_boot, 
                                    mean_shear_error, mean_GSN_shear_error, weighted_mean_shear, weighted_mean_shear_error, 
                                    weighted_GSN_mean_shear, weighted_GSN_mean_shear_error)), 
                           fmt=' '.join(['%2.7f']*11), 
                           header=header, 
                           comments='')
                
            # Save inverse Hartlap-corrected covariance matrix
            inverse_cov_filename = pre_path + '/output_relative/' + f'{cosmoin[i]}_inverse_cov_H.dat'
            header = 'Inverse covariance matrix 64x64 corrected with Hartlap method'
            np.savetxt(inverse_cov_filename, inverse_cov_H, fmt='%2.10f', header=header, comments='')

                
def process_shear_multibinsize(pre_path, cosmoin, dirs, n, nbins):
    import Finder_functions as mystery
    rp = None
    subfolderss = [str(l_c).zfill(2) for l_c in range(n)]
    
    for i in range(len(cosmoin)):
        print()
        print(f'{cosmoin[i]}:')
        if rp is None and i == 0:
            path_rp = pre_path + '/output_relative/' + cosmoin[i] + "/" + subfolderss[0] + "/" + dirs[0] + "/"
            radii = list(np.genfromtxt(path_rp + 'void_0.txt', usecols=(0), unpack=True, skip_header=2))        
            rp = np.array(radii)
        
        for ii in range(len(dirs)):
            folderout = pre_path + "/output_relative/"
            bin_counter = 0
            shear__bin = []
            shear_errors_bin = []
            GSN_errors_bin = []
            print(dirs[ii])
    
            for l_c in subfolderss:
                folderin = pre_path + '/output_relative/' + cosmoin[i] + "/" + l_c + "/"
                path = folderin + dirs[ii] + '/'

                file_list = [file for file in glob.glob(path + '*') if not file.endswith('_log.txt')]
                file_list = sorted(file_list, key=lambda x: int(re.search(r'void_(\d+)\.txt', x).group(1)))

                for filename in file_list:
                    _, kappa, kappa_shell, GSN_shell, shear_error_shell = np.genfromtxt(filename, usecols=(0, 1, 2, 6, 7), unpack=True, skip_header=2)
                    shear = kappa - kappa_shell
                    var = True
                    
                    if var:
                        bin_counter += 1
                        shear__bin.append(shear)
                        GSN_errors_bin.append(GSN_shell) # # Add errors from galaxy shape noise
                        shear_errors_bin.append(shear_error_shell)  # Add errors from measure
            
            shear_bin = np.array(shear__bin)
            GSN_errors_bin = np.array(GSN_errors_bin)
            shear_errors_bin = np.array(shear_errors_bin)
        
            #if shear_map.ndim == 1:
                #shear_bin = shear_bin[np.newaxis, :]
                #GSN_errors_bin = GSN_errors_bin[np.newaxis, :]
                #shear_errors_bin = shear_errors_bin[np.newaxis, :]             

            bin_mean_shear = np.mean(shear_bin, axis=0)
            bin_std_shear = np.std(shear_bin, axis=0) / np.sqrt(bin_counter)

            mean_shear_error = np.sqrt(np.sum(shear_errors_bin**2, axis=0)) / bin_counter
            mean_GSN_shear_error = np.sqrt(np.sum(GSN_errors_bin**2, axis=0)) / bin_counter

            weights = 1 / shear_errors_bin**2
            weighted_mean_shear = np.sum(shear_bin * weights, axis=0) / np.sum(weights, axis=0)

            weights_GSN = 1 / GSN_errors_bin**2
            weighted_GSN_mean_shear = np.sum(shear_bin * weights_GSN, axis=0) / np.sum(weights_GSN, axis=0)

            weighted_mean_shear_error = 1 / np.sqrt(np.sum(weights, axis=0))
            weighted_GSN_mean_shear_error = 1 / np.sqrt(np.sum(weights_GSN, axis=0))

            bin_shear_error_cov, bin_shear_error_jack, bin_shear_error_boot, bin_shear_errror_H, bin_inverse_cov_H = mystery.calculate_stat_error(shear_bin, nbins, 1000)

            bin_shear_error_cov = mystery.round_up_decimal(bin_shear_error_cov, 7)
            bin_shear_error_jack = mystery.round_up_decimal(bin_shear_error_jack, 7)
            bin_shear_error_boot = mystery.round_up_decimal(bin_shear_error_boot, 7)

            bin_mean_shear = mystery.round_up_decimal(bin_mean_shear, 7)
            bin_std_shear = mystery.round_up_decimal(bin_std_shear, 7)

            mean_shear_error = mystery.round_up_decimal(mean_shear_error, 7)
            mean_GSN_shear_error = mystery.round_up_decimal(mean_GSN_shear_error, 7)

            weighted_mean_shear = mystery.round_up_decimal(weighted_mean_shear, 7)
            weighted_mean_shear_error = mystery.round_up_decimal(weighted_mean_shear_error, 7)

            weighted_GSN_mean_shear = mystery.round_up_decimal(weighted_GSN_mean_shear, 7)
            weighted_GSN_mean_shear_error = mystery.round_up_decimal(weighted_GSN_mean_shear_error, 7)

            print(bin_counter)
            
            # Save for binsize
            bin_filename = folderout + f'{cosmoin[i]}_binsize_{ii+1}_mean_shear.dat'
            header = 'r/R_v  gt_MEAN  ERROR_COVMAT  ERROR_JACKKNIFE  ERROR_BOOTSTRAP  ESM_Kmeasure  ESM_GSN  gt_WEIGHTED_Kmeasure  ERROR_gt_WEIGHTED_Kmeasure  gt_WEIGHTED_GSN  ERROR_gt_WEIGHTED_GSN'

            with open(bin_filename, 'w') as file:
                np.savetxt(file,
                           list(zip(rp, bin_mean_shear, bin_shear_error_cov, bin_shear_error_jack, bin_shear_error_boot, 
                                    mean_shear_error, mean_GSN_shear_error, weighted_mean_shear, weighted_mean_shear_error, 
                                    weighted_GSN_mean_shear, weighted_GSN_mean_shear_error)), 
                           fmt=' '.join(['%2.7f']*11), 
                           header=header, 
                           comments='')
                
            # Save inverse Hartlap-corrected covariance matrix
            bin_inverse_cov_filename = folderout + f'{cosmoin[i]}_binsize_{ii+1}_inverse_cov_H.dat'
            header = 'Inverse covariance matrix 64x64 corrected with Hartlap method'
            np.savetxt(bin_inverse_cov_filename, bin_inverse_cov_H, fmt='%2.10f', header=header, comments='')


def plot_mean_shear(pre_path, cosmoin, fradius, line_style, color, binsize=0):
    folderout = pre_path + '/output_relative'
    
    if binsize==0:
        rp, mean_shear_lcdm, errors_lcdm = np.genfromtxt(folderout + f'/{cosmoin[0]}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        mean_shear_fR4, errors_fR4 = np.genfromtxt(folderout + f'/{cosmoin[1]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR5, errors_fR5 = np.genfromtxt(folderout + f'/{cosmoin[2]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR6, errors_fR6 = np.genfromtxt(folderout + f'/{cosmoin[3]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_lcdm15, errors_lcdm15 = np.genfromtxt(folderout + f'/{cosmoin[4]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
    else:
        rp, mean_shear_lcdm, errors_lcdm = np.genfromtxt(folderout + f'/{cosmoin[0]}_binsize_{binsize}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        mean_shear_fR4, errors_fR4 = np.genfromtxt(folderout + f'/{cosmoin[1]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR5, errors_fR5 = np.genfromtxt(folderout + f'/{cosmoin[2]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR6, errors_fR6 = np.genfromtxt(folderout + f'/{cosmoin[3]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_lcdm15, errors_lcdm15 = np.genfromtxt(folderout + f'/{cosmoin[4]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
    
    # Significance (sigma)
    sigma_lcdm = np.sum((mean_shear_lcdm**2) / (errors_lcdm**2))
    print(sigma_lcdm)
    sigma_fR4 = np.sum((mean_shear_fR4**2) / (errors_fR4**2))
    sigma_fR5 = np.sum((mean_shear_fR5**2) / (errors_fR5**2))
    sigma_fR6 = np.sum((mean_shear_fR6**2) / (errors_fR6**2))
    sigma_lcdm15 = np.sum((mean_shear_lcdm15**2) / (errors_lcdm15**2))
    
    cosmolabels = []
    for cosmo in cosmoin:
        if cosmo == 'LCDM':
            cosmolabels.append(r'$\Lambda$CDM')
        elif cosmo == 'LCDM_0.15':
            cosmolabels.append(r'$\Lambda$CDM$_{0.15 \, eV}$')
        elif cosmo == 'fR4':
            cosmolabels.append(r'$f$R$4$')
        elif cosmo == 'fR5':
            cosmolabels.append(r'$f$R$5$')
        elif cosmo == 'fR6':
            cosmolabels.append(r'$f$R$6$')

    # Plot cosmology
    plt.figure(figsize=(18, 14))
    plt.xlim(0, fradius * 0.5)
    plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1)) 
    plt.ylim(-7.5, 0.5)
    plt.xlabel('$r_p/R_v$', fontsize=40)
    plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=40)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    
    if binsize==0:
        plt.errorbar(rp, mean_shear_lcdm*1e3, yerr=errors_lcdm*1e3, label=cosmolabels[0], linestyle=line_style[0], color=color[0])
        plt.plot(rp, mean_shear_fR4*1e3, label=cosmolabels[1], linestyle=line_style[1], color=color[1])
        plt.plot(rp, mean_shear_fR5*1e3, label=cosmolabels[2], linestyle=line_style[2], color=color[2])
        plt.plot(rp, mean_shear_fR6*1e3, label=cosmolabels[3], linestyle=line_style[3], color=color[3])
        plt.plot(rp, mean_shear_lcdm15*1e3, label=cosmolabels[4], linestyle=line_style[4], color=color[4])
    else:
        bin_edges = [1, 3.79, 4.85, 15]
        min_radius = bin_edges[binsize-1]
        max_radius = bin_edges[binsize]
        labelbin = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'
        
        plt.errorbar(rp, mean_shear_lcdm*1e3, yerr=errors_lcdm*1e3, label=f'{cosmolabels[0]} {labelbin}', linestyle=line_style[0], color=color[0])
        plt.plot(rp, mean_shear_fR4*1e3, label=f'{cosmolabels[1]} {labelbin}', linestyle=line_style[1], color=color[1])
        plt.plot(rp, mean_shear_fR5*1e3, label=f'{cosmolabels[2]} {labelbin}', linestyle=line_style[2], color=color[2])
        plt.plot(rp, mean_shear_fR6*1e3, label=f'{cosmolabels[3]} {labelbin}', linestyle=line_style[3], color=color[3])
        plt.plot(rp, mean_shear_lcdm15*1e3, label=f'{cosmolabels[4]} {labelbin}', linestyle=line_style[4], color=color[4])

    plt.legend(fontsize=40)
    
    return sigma_lcdm, sigma_fR4, sigma_fR5, sigma_fR6, sigma_lcdm15

def plot_differences_LCDM(pre_path, cosmoin, fradius, line_style, color, fb, binsize=0, only=False):
    folderout = pre_path + '/output_relative'
    
    if binsize==0:
        rp, mean_shear_lcdm, errors_lcdm = np.genfromtxt(folderout + f'/{cosmoin[0]}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        mean_shear_fR4, errors_fR4 = np.genfromtxt(folderout + f'/{cosmoin[1]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR5, errors_fR5 = np.genfromtxt(folderout + f'/{cosmoin[2]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR6, errors_fR6 = np.genfromtxt(folderout + f'/{cosmoin[3]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_lcdm15, errors_lcdm15 = np.genfromtxt(folderout + f'/{cosmoin[4]}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
    else:
        rp, mean_shear_lcdm, errors_lcdm = np.genfromtxt(folderout + f'/{cosmoin[0]}_binsize_{binsize}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        mean_shear_fR4, errors_fR4 = np.genfromtxt(folderout + f'/{cosmoin[1]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR5, errors_fR5 = np.genfromtxt(folderout + f'/{cosmoin[2]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_fR6, errors_fR6 = np.genfromtxt(folderout + f'/{cosmoin[3]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
        mean_shear_lcdm15, errors_lcdm15 = np.genfromtxt(folderout + f'/{cosmoin[4]}_binsize_{binsize}_mean_shear.dat', usecols=(1, 2), unpack=True, skip_header=1)
    
    cosmolabels = []
    for cosmo in cosmoin:
        if cosmo == 'LCDM':
            cosmolabels.append(r'$\Lambda$CDM')
        elif cosmo == 'LCDM_0.15':
            cosmolabels.append(r'$\Lambda$CDM$_{0.15 \, eV}$')
        elif cosmo == 'fR4':
            cosmolabels.append(r'$f$R$4$')
        elif cosmo == 'fR5':
            cosmolabels.append(r'$f$R$5$')
        elif cosmo == 'fR6':
            cosmolabels.append(r'$f$R$6$')
    
    if only==False:
        fig, axs = plt.subplots(1, 2, figsize=(36, 14))

        # Plot mean shear
        plt.sca(axs[0])
        plt.ylim(-8, 0.5)
        plt.xlim(0, fradius * 0.5)
        plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1), fontsize=35)
        plt.yticks(fontsize=35)
        plt.xlabel('$r_p/R_v$', fontsize=40)
        plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=40)

        if binsize != 0:
            bin_edges = [1, 3.79, 4.85, 15]
            min_radius = bin_edges[binsize-1]
            max_radius = bin_edges[binsize]
            labelbin = f'$R_v$ in [{min_radius}-{max_radius}] arcmin'
        else:
            labelbin = ''

        plt.errorbar(rp, mean_shear_lcdm*1e3, yerr=errors_lcdm*1e3, label=f'{cosmolabels[0]} {labelbin}', linestyle=line_style[0], color=color[0])
        plt.plot(rp, mean_shear_fR4*1e3, label=f'{cosmolabels[1]} {labelbin}', linestyle=line_style[1], color=color[1])
        plt.plot(rp, mean_shear_fR5*1e3, label=f'{cosmolabels[2]} {labelbin}', linestyle=line_style[2], color=color[2])
        plt.plot(rp, mean_shear_fR6*1e3, label=f'{cosmolabels[3]} {labelbin}', linestyle=line_style[3], color=color[3])
        plt.plot(rp, mean_shear_lcdm15*1e3, label=f'{cosmolabels[4]} {labelbin}', linestyle=line_style[4], color=color[4])

        plt.legend(fontsize=40)


    # Define the values of the ordinates for the differences relative to ΛCDM
    ddiff0 = (mean_shear_lcdm - mean_shear_lcdm)/abs(mean_shear_lcdm)
    ddiff4 = (mean_shear_fR4 - mean_shear_lcdm)/abs(mean_shear_lcdm)
    ddiff5 = (mean_shear_fR5 - mean_shear_lcdm)/abs(mean_shear_lcdm)
    ddiff6 = (mean_shear_fR6 - mean_shear_lcdm)/abs(mean_shear_lcdm)
    ddiff0_15 = (mean_shear_lcdm15 - mean_shear_lcdm)/abs(mean_shear_lcdm)

    # Calculate the minimum and maximum of the relative differences
    ddiff_min = np.minimum.reduce([np.min(ddiff0), np.min(ddiff4), np.min(ddiff5), np.min(ddiff6), np.min(ddiff0_15)]) - 0.03
    ddiff_max = np.maximum.reduce([np.max(ddiff0), np.max(ddiff4), np.max(ddiff5), np.max(ddiff6), np.max(ddiff0_15)]) + 0.03

    diff_err_lcdm = errors_lcdm/abs(mean_shear_lcdm)
    diff_err_4 = errors_fR4/abs(mean_shear_lcdm)
    diff_err_5 = errors_fR5/abs(mean_shear_lcdm)
    diff_err_6 = errors_fR6/abs(mean_shear_lcdm)
    diff_err_15 = errors_lcdm15/abs(mean_shear_lcdm)

    if only==False:
        plt.sca(axs[1])        
    else:
        plt.figure(figsize=(14, 14))
        
    plt.xlim(0, fradius * 0.5)
    plt.ylim(ddiff_min, ddiff_max)
    plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1), fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel('$r_p/R_v$', fontsize=40)
    plt.ylabel(r'$(\gamma_t^{{\rm i}} - \gamma_t^{{\rm i, \Lambda CDM}})/|\gamma_t^{{\rm i, \Lambda CDM}}|$', fontsize=40)

    rr_ddiff = np.linspace(0, (fradius*0.5), int(len(rp)))

    if binsize != 0:
        bin_edges = [1, 3.79, 4.85, 15]
        min_radius = bin_edges[binsize-1]
        max_radius = bin_edges[binsize]
        labelbin = f'$R_v$ in [{min_radius}-{max_radius}] arcmin'
    else:
        labelbin = ''

    plt.errorbar(rr_ddiff, ddiff0, yerr=diff_err_lcdm, linestyle=line_style[0], color=color[0], label=f'{cosmolabels[0]} {labelbin}', linewidth=1, zorder=1 if fb else 10)
    plt.plot(rr_ddiff, ddiff4, linestyle=line_style[1], color=color[1], label=f'{cosmolabels[1]} {labelbin}', linewidth=3.5, zorder=10)
    plt.plot(rr_ddiff, ddiff5, linestyle=line_style[2], color=color[2], label=f'{cosmolabels[2]} {labelbin}', linewidth=3.5, zorder=10)
    plt.plot(rr_ddiff, ddiff6, linestyle=line_style[3], color=color[3], label=f'{cosmolabels[3]} {labelbin}', linewidth=3.5, zorder=10)
    plt.plot(rr_ddiff, ddiff0_15, linestyle=line_style[4], color=color[4], label=f'{cosmolabels[4]} {labelbin}', linewidth=3.5, zorder=10)

    if fb:
        plt.fill_between(rr_ddiff, ddiff4 - diff_err_4, ddiff4 + diff_err_4, color=color[1], alpha=0.3)
        plt.fill_between(rr_ddiff, ddiff5 - diff_err_5, ddiff5 + diff_err_5, color=color[2], alpha=0.3)
        plt.fill_between(rr_ddiff, ddiff6 - diff_err_6, ddiff6 + diff_err_6, color=color[3], alpha=0.3)
        plt.fill_between(rr_ddiff, ddiff0_15 - diff_err_15, ddiff0_15 + diff_err_15, color=color[4], alpha=0.3)

    plt.legend(fontsize=40)
        
    if only==False:
        plt.savefig("plot_complete.pdf")
    else:
        plt.savefig("plot_res.pdf")
    
def plot_bins(pre_path, cosmoc, dirs, fradius, line_style, color_dir, fb):
    folderout = pre_path + '/output_relative'
            
    if cosmoc == 'LCDM':
        cosmolabel = '$\Lambda$CDM'
    elif cosmoc == 'LCDM_0.15':
        cosmolabel = '$\Lambda$CDM$_{0.15 \, eV}$'
    elif cosmoc == 'fR4':
        cosmolabel = '$f$R$5$'
    elif cosmoc == 'fR5':
        cosmolabel = '$f$R$4$'
    elif cosmoc == 'fR6':
        cosmolabel = '$f$R$6$'        

    fig, axs = plt.subplots(1, 2, figsize=(36, 14))
    
    # Plot bins mean shear
    plt.sca(axs[0])
    plt.ylim(-8.5, 0.5)
    plt.xlim(0, fradius * 0.5)
    plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1), fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel('$r_p/R_v$', fontsize=40)
    plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=40)

    rp, mean_shear, errors = np.genfromtxt(folderout + f'/{cosmoc}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
    plt.errorbar(rp, mean_shear*1e3, yerr=errors*1e3, label=r'$\overline{\gamma_t}(r_p)$'fr' {cosmolabel}', linestyle=line_style[1], color='k', linewidth=2.5)
    diff_mean_shear = (mean_shear - mean_shear)/abs(mean_shear)
    differr = errors/abs(mean_shear)
    
    diff_stack0 = []
    diff_stack1 = []
    diff_stack2 = []
    differr_stack0 = []
    differr_stack1 = []
    differr_stack2 = []    
    
    bin_edges = [1, 3.79, 4.85, 15]    
    for i in range(len(dirs)):
        min_radius = bin_edges[i]
        max_radius = bin_edges[i+1]
        labelbin = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'

        rp, mean_shear_bin, errors_bin = np.genfromtxt(folderout + f'/{cosmoc}_binsize_{i+1}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        plt.errorbar(rp, mean_shear_bin*1e3, yerr=errors_bin*1e3, label=f'{labelbin}', linestyle=line_style[0], color=color_dir[i], linewidth=2.5)
        diff_stack = (mean_shear_bin - mean_shear)/abs(mean_shear)
        differr_stack = errors_bin/abs(mean_shear)
        if i == 0:
            diff_stack0 = diff_stack
            differr_stack0 = differr_stack
        elif i == 1:
            diff_stack1 = diff_stack
            differr_stack1 = differr_stack
        elif i == 2:
            diff_stack2 = diff_stack
            differr_stack2 = differr_stack    
            
    plt.legend(fontsize=40)

    diff_min_s = np.minimum.reduce([np.min(diff_mean_shear), np.min(diff_stack0), np.min(diff_stack1), np.min(diff_stack2)]) - 0.03
    diff_max_s = np.maximum.reduce([np.max(diff_mean_shear), np.max(diff_stack0), np.max(diff_stack1), np.max(diff_stack2)]) + 0.03

    # Plot bins differences
    plt.sca(axs[1])
    plt.xlim(0, fradius * 0.5)
    plt.ylim(diff_min_s, diff_max_s)
    plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1), fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel('$r_p/R_v$', fontsize=40)
    if cosmolabel == '$\Lambda$CDM':
        plt.ylabel(r'$(\gamma_t^{{\rm i}} - \gamma_t^{{\rm i, \Lambda CDM}})/|\gamma_t^{{\rm i, \Lambda CDM}}|$', fontsize=40)
    elif cosmolabel == '$\Lambda$CDM$_{0.15 \, eV}$':
        plt.ylabel(r'$(\gamma_t^{{\rm i}} - \gamma_t^{{\rm i, \Lambda CDM_{0.15 \,\, eV}}})/|\gamma_t^{{\rm i, \Lambda CDM_{0.15 \,\, eV}}}|$', fontsize=40)
    else:
        plt.ylabel(r'$(\gamma_t^{i} - \gamma_t^{i, ' + cosmolabel + '})/|\gamma_t^{i, ' + cosmolabel + '}|$', fontsize=40)
    plt.sca(axs[1])

    rr_diff = np.linspace(0, (fradius*0.5), int(len(rp)))

    plt.errorbar(rr_diff, diff_mean_shear, yerr=differr, linestyle=line_style[1], color='k', label=r'$\overline{\gamma_t}(r_p)$'fr' {cosmolabel}', linewidth=2.5)

    plt.plot(rr_diff, diff_stack0, color=color_dir[0], label='$R_v$ in ' + '[' + str(bin_edges[0]) + '-' + str(bin_edges[1]) + '] arcmin', linewidth=2.5)
    plt.plot(rr_diff, diff_stack1, color=color_dir[1], label='$R_v$ in ' + ']' + str(bin_edges[1]) + '-' + str(bin_edges[2]) + '] arcmin', linewidth=2.5)
    plt.plot(rr_diff, diff_stack2, color=color_dir[2], label='$R_v$ in ' + ']' + str(bin_edges[2]) + '-' + str(bin_edges[3]) + '] arcmin', linewidth=2.5)

    if fb:
        plt.fill_between(rr_diff, diff_stack0 - differr_stack0, diff_stack0 + differr_stack0, color=color_dir[0], alpha=0.3)
        plt.fill_between(rr_diff, diff_stack1 - differr_stack1, diff_stack1 + differr_stack1, color=color_dir[1], alpha=0.3)
        plt.fill_between(rr_diff, diff_stack2 - differr_stack2, diff_stack2 + differr_stack2, color=color_dir[2], alpha=0.3)

    plt.legend(fontsize=40)
    #plt.savefig("plot_bins.pdf")

    
def plot_every_bin(pre_path, cosmoin, dirs, fradius, line_style, color_dir):
    folderout = pre_path + '/output_relative'
    bin_edges = [1, 3.79, 4.85, 15]
    
    plt.rcParams['figure.figsize'] = [18, 14]
    plt.ylim(-8.5,0.5)
    plt.xlim(0,fradius*0.5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=3)
    plt.xticks(np.arange(0, (fradius*0.5)+1, step=1), fontsize=35)
    plt.yticks(fontsize=35)
    plt.xlabel('$r_p/R_v$', fontsize=40)
    plt.ylabel(fr'$\gamma_t(r_p) \, [10^{{-3}}]$', fontsize=40, labelpad=15)
    plt.tick_params(axis='both', which='major', direction='in', width=2)
    # Imposta bordi neri più spessi
    ax = plt.gca()  # Ottieni l'asse corrente
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    for cc_run in range(len(cosmoin)):
        for i in range(len(dirs)):
            min_radius = bin_edges[i]
            max_radius = bin_edges[i+1]
            rp, mean_shear, errors = np.genfromtxt(folderout + f'/{cosmoin[cc_run]}_binsize_{i+1}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
            plt.errorbar(rp, mean_shear*1e3, errors*1e3, linestyle=line_style[cc_run], color=color_dir[i], label='$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin', linewidth=2.5)
    
    legend_elements = []
    for i in range(len(dirs)):
        min_radius = bin_edges[i]
        max_radius = bin_edges[i+1]
        if i==0:
            label = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'
        else:
            label = '$R_v$ in ' + ']' + str(min_radius) + '-' + str(max_radius) + '] arcmin'
        line = plt.Line2D([], [], color=color_dir[i], label=label)
        legend_elements.append(line)

    cosmolabels = []
    for cosmo in cosmoin:
        if cosmo == 'LCDM':
            cosmolabels.append(r'$\Lambda$CDM')
        elif cosmo == 'LCDM_0.15':
            cosmolabels.append(r'$\Lambda$CDM$_{0.15 \, eV}$')
        elif cosmo == 'fR4':
            cosmolabels.append(r'$f$R$4$')
        elif cosmo == 'fR5':
            cosmolabels.append(r'$f$R$5$')
        elif cosmo == 'fR6':
            cosmolabels.append(r'$f$R$6$')
    
    for cc_run in range(len(cosmolabels)):
        label = cosmolabels[cc_run]
        line = plt.Line2D([], [], color='k', label=label, linestyle=line_style[cc_run])
        legend_elements.append(line)
    
    plt.legend(handles=legend_elements, fontsize=40)
    plt.savefig("/home/leonardo/Desktop/pdfs2/all_bins_cosmo.pdf", format='pdf', bbox_inches='tight')
    
                
def extract_cobaya(chain_root, cobaya_dir, names, n_cores, n_cols):
    concatenated_array = np.empty((len(names), 0)).tolist()
    for num in range(1,n_cores+1):
        pars = np.genfromtxt(cobaya_dir+chain_root+"."+str(num)+".txt", usecols=n_cols)
        concatenated_array = np.concatenate((concatenated_array, pars.T), axis=1)
    return concatenated_array.T

def prepare_sample(samples, plot_label, names, labels, burn_in=0.1):
    return MCSamples(samples=samples, label=plot_label, names=names, labels=labels, ignore_rows=burn_in)

def prepare_plot(sigmas=2): 
    g = plots.getSubplotPlotter(width_inch=14)
    
    # Impostazioni di default
    g.settings.axes_fontsize = 17
    g.settings.lab_fontsize = 17
    g.settings.legend_fontsize = 14
    g.settings.lw_contour = sigmas
    g.settings.legend_loc = 'lower left'
    
    # Generiamo un plot temporaneo per recuperare i livelli di contorno effettivi
    temp_plot = plots.GetDistPlotter()
    temp_plot.settings.num_plot_contours = sigmas  # Usiamo n livelli di default di GetDist
    temp_levels = temp_plot.settings.num_plot_contours  # GetDist ci dice quanti usa di default

    print(f"Livelli di confidenza reali usati da GetDist: {temp_levels}")

    contour_args = {}

    #if sigmas:
        # Recuperiamo i livelli di confidenza effettivi
        #confidence_levels = np.linspace(0.6827, 0.9999994, len(sigmas))
        #print(f"Livelli di confidenza uniformati a GetDist: {confidence_levels}")
        
        # Normalizziamo per allinearci al metodo di GetDist
        #max_conf = 1.0
        #confidence_levels = [level / max_conf for level in confidence_levels]
        
        #contour_args['contour_levels'] = confidence_levels
        #print(f"Contour levels normalizzati: {contour_args['contour_levels']}")
    
    return g, contour_args


def new_function(rp, a, b, c, d, e):
    shear_model_fit = rp * a * (((1 - rp**b) / (1 + rp**c)) - (np.exp(d * rp) / (1 + np.exp(e*rp - (d+e)))))
    return shear_model_fit.flatten()

def calcola_delta_unit_stack_N(R_l, iRv, rp, rs, alpha, beta, rho_m):
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    interp_yy_nn = []
    for num in R_l:
        def integrand(rz):
            r = np.sqrt((rz)**2 + num**2)
            return ((1 - (r / rs)**alpha) / (1 + (r)**beta))

        result, _ = quad(integrand, 0.001, 10, limit=100000, epsrel=1e-4)

        interp_yy_nn.append(result*2)
    sigma_rp_interp_nn = interp1d(R_l, interp_yy_nn, kind='quadratic', fill_value='extrapolate')

    sigma_rp_enclosed_values_nn = []
    for nn in range(len(rp)): #ciclo su tutti i raggi proiettati
        def sigma_enclosed(rp_prime):
            result = 2 * np.pi * rp_prime * sigma_rp_interp_nn(rp_prime)
            return result

        results = (1 / (np.pi*(rp[nn]**2)))*quad(sigma_enclosed, 0.001, rp[nn], limit=100000, epsrel=1e-5)[0]
        sigma_rp_enclosed_values_nn.append(results)
    sigma_rp_enclosed_values_nn = np.array(sigma_rp_enclosed_values_nn)
    delta_sigma_values_unitless_nn = sigma_rp_enclosed_values_nn - sigma_rp_interp_nn(rp)
    delta_sigma_values_unit_nn = delta_sigma_values_unitless_nn * iRv * rho_m * (1/1e12) #* delta_c
    return delta_sigma_values_unit_nn, sigma_rp_enclosed_values_nn, sigma_rp_interp_nn(rp)

def calculate_shear_model_fit_N(rp, r_s_over_R_vv, alphaa, betaa, delta_prime):
    import Finder_functions as mystery
    rho_m = 8.699385581016718*1e10
    R_l = np.linspace(1e-4, 1e1, 1000)
    Rv_peak = 1.2500443529452991
    delta_senza_dc, _, _ = mystery.calcola_delta_unit_stack_N(R_l, Rv_peak, rp, r_s_over_R_vv, alphaa, betaa, rho_m)
    shear_model_fit = delta_senza_dc * delta_prime
    return shear_model_fit.flatten()

def calcola_delta_unit_stack_T(R_l, rp, s, iRv, rho_m, rangeint):
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    interp_yy_nn = []
    for num in R_l: #ciclo su tutti i raggi proiettati della griglia
        def H_integrand(rz):
            r = np.sqrt((rz)**2 + num**2)
            y = np.log(r/iRv)          #adimensional
            r0 = 0.37*s**2 + 0.25*s + 0.89  #in Mpc/h
            y0 = np.log(r0/iRv)        #adimensional
            return (0.5 * (1 + np.tanh((y-y0)/s)) - 1)  #* np.abs(delta_cc)

        result, _ = quad(H_integrand, 0., rangeint, limit=100000, epsrel=1e-4)

        interp_yy_nn.append(result*2)
    sigma_rp_interp_nn = interp1d(R_l, interp_yy_nn, kind='quadratic', fill_value='extrapolate')

    sigma_rp_enclosed_values_nn = []

    for nn in range(len(rp)): #ciclo su tutti i raggi proiettati
        def sigma_enclosed(rp_prime):
            result = 2 * np.pi * rp_prime * sigma_rp_interp_nn(rp_prime)
            return result

        results = (1 / (np.pi*(rp[nn]**2)))*quad(sigma_enclosed, 0., rp[nn], limit=100000, epsrel=1e-5)[0]
        sigma_rp_enclosed_values_nn.append(results)
    sigma_rp_enclosed_values_nn = np.array(sigma_rp_enclosed_values_nn)
    delta_sigma_values_unitless_nn = sigma_rp_enclosed_values_nn - sigma_rp_interp_nn(rp)
    delta_sigma_values_unit_nn = delta_sigma_values_unitless_nn * iRv * rho_m * (1/1e12)
    return delta_sigma_values_unit_nn, sigma_rp_enclosed_values_nn, sigma_rp_interp_nn(rp)

def calculate_shear_model_fit_T(rp, ss, delta_prime):
    import Finder_functions as mystery
    rangeint = 1147.9660518953947 #D_z/2
    rho_m = 8.699385581016718*1e10
    Rv_peak = 1.2500443529452991
    R_l = np.linspace(1e-4, 1e1, 1000)
    delta_senza_dc, _, _ = mystery.calcola_delta_unit_stack_T(R_l, rp, ss, Rv_peak, rho_m, rangeint)
    shear_model_fit = delta_senza_dc * np.abs(delta_prime)
    return shear_model_fit.flatten()

def best_fit_model_mcmc(pre_path, cosmoin, n_cores, cobaya_path, fradius, binsize=0):
    import Finder_functions as mystery
    from tqdm import tqdm
    # Caricamento dati shear
    if binsize==0:
        rp, mean_shear, mean_shear_error = np.genfromtxt(pre_path + '/output_relative/' + f'{cosmoin}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
    else:
        rp, mean_shear, mean_shear_error = np.genfromtxt(pre_path + '/output_relative/' + f'{cosmoin}_binsize_{binsize}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        
    rp = rp[1:]
    mean_shear = mean_shear[1:]
    mean_shear_error = mean_shear_error[1:]
    
    if cobaya_path == '/home/leonardo/Desktop/DUSTGRAIN-pathfinder/New_Maggiore_profile/':
        cobaya_chains_name = "cobaya_N_M_Profile"
        fitting_function = mystery.new_function
        names = ['a', 'b', 'c', 'd', 'e']
        n_cols = (2, 3, 4, 5, 6)
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin}/chains/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin}_binsize_{binsize}/chains/'
            
    elif cobaya_path == '/home/leonardo/Desktop/DUSTGRAIN-pathfinder/N_profile/':
        cobaya_chains_name = "cobaya_N_Profile"
        fitting_function = mystery.calculate_shear_model_fit_N
        names = ['rs/R_v', '\\alpha', '\\beta', '\delta']
        n_cols = (2, 3, 4, 5)
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin}/chains/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin}_binsize_{binsize}/chains/'
    
    elif cobaya_path == '/home/leonardo/Desktop/DUSTGRAIN-pathfinder/T_profile/':
        cobaya_chains_name = "cobaya_T_Profile"
        fitting_function = mystery.calculate_shear_model_fit_T
        names = ['s', '\delta']
        n_cols = (2,3)
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin}/chains/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin}_binsize_{binsize}/chains/'
            
    samples0 = mystery.extract_cobaya(cobaya_chains_name, cobaya_dir, names, n_cores, n_cols)
    #print (np.cov(samples0.T))
    
    # Calculate model's best-fit
    best_fit = np.median(samples0, axis=0)
    #print(fr'Best Fit for {cosmoin}:', best_fit)

    num_row = samples0.shape[0]
    num_params = samples0.shape[1]

    # Percentiles:
    percentiles_1std = [15.865, 50, 84.135]  # 1 (68,27%) [0, 1, 2]
    percentiles_2std = [2.275, 50, 97.725]   # 2 (95,45%)
    percentiles_3std = [0.15, 50, 99.85]     # 3 (99,73%)

    # Calculate confidence intervals for every percentiles
    param_fitted_1std, param_errors_1std = mystery.calculate_param_intervals(samples0, percentiles_1std, num_params)
    param_fitted_2std, param_errors_2std = mystery.calculate_param_intervals(samples0, percentiles_2std, num_params)
    param_fitted_3std, param_errors_3std = mystery.calculate_param_intervals(samples0, percentiles_3std, num_params)

    # Calculate fitted model
    x_data = rp
    y_data_median_discreta = fitting_function(x_data, *best_fit)
    
    rppp = np.linspace(0, x_data.max(), 100000) # smoothed x
    y_data_median = fitting_function(rppp, *best_fit)
    
    matrix = np.zeros((num_row, len(x_data)))

    for i in tqdm(range(num_row), desc="Processing rows", unit="row"):
        # Obtain current parameters
        params = samples0[i, :]

        # Calculate model for current parameters
        model = fitting_function(x_data, *params)

        # Adding model
        matrix[i, :] = model

        # Plot model
        ###plt.plot(x_data, model*1e3, color='gray', alpha=0.1)    

    # Bounds
    lower_bounds_1std = np.zeros_like(x_data)
    upper_bounds_1std = np.zeros_like(x_data)

    lower_bounds_2std = np.zeros_like(x_data)
    upper_bounds_2std = np.zeros_like(x_data)

    lower_bounds_3std = np.zeros_like(x_data)
    upper_bounds_3std = np.zeros_like(x_data)

    for ii in range(len(x_data)):
        column_vector = matrix[:, ii]
        #print(f"Column {ii + 1}:", column_vector)
        column_vector = np.array(column_vector)

        # Calculate confidence intervals at 1, 2 and 3 sigma
        interval_1std = np.percentile(column_vector, percentiles_1std)
        interval_2std = np.percentile(column_vector, percentiles_2std)
        interval_3std = np.percentile(column_vector, percentiles_3std)

        lower_bounds_1std[ii] = interval_1std[0]
        upper_bounds_1std[ii] = interval_1std[2]

        lower_bounds_2std[ii] = interval_2std[0]
        upper_bounds_2std[ii] = interval_2std[2]    

        lower_bounds_3std[ii] = interval_3std[0]
        upper_bounds_3std[ii] = interval_3std[2]

    # Degrees of freedom
    df = len(x_data) - num_params

    # Residuals
    residuals = mean_shear - y_data_median_discreta

    # Calculate reduced chi square of the fit
    fit_reduced_chi_square = np.sum(((residuals / mean_shear_error) ** 2) / df)

    print(f"Model reduced Chi square: {fit_reduced_chi_square:.7f}")
    print()
    
    if fitting_function == mystery.new_function:
        a_median, b_median, c_median, d_median, e_median = best_fit
        
        print("Best fit parameters:")
        print(f"a: {a_median:.7f} +/- {param_errors_1std[0]:.7f}")
        print(f"b: {b_median:.7f} +/- {param_errors_1std[1]:.7f}")
        print(f"c: {c_median:.7f} +/- {param_errors_1std[2]:.7f}")
        print(f"d: {d_median:.7f} +/- {param_errors_1std[3]:.7f}")
        print(f"e: {e_median:.7f} +/- {param_errors_1std[4]:.7f}")
        
    elif fitting_function == mystery.calculate_shear_model_fit_N:
        rs_median, alpha_median, beta_median, delta_median = best_fit
        
        print("Best fit parameters:")
        print(f"rs/R_v: {rs_median:.7f} +/- {param_errors_1std[0]:.7f}")
        print(f"\\alpha: {alpha_median:.7f} +/- {param_errors_1std[1]:.7f}")
        print(f"\\beta: {beta_median:.7f} +/- {param_errors_1std[2]:.7f}")
        print(f"\delta: {delta_median:.7f} +/- {param_errors_1std[3]:.7f}")
    
    elif fitting_function == mystery.calculate_shear_model_fit_T:
        s_median, delta_median = best_fit
        
        print("Best fit parameters:")
        print(f"s: {s_median:.7f} +/- {param_errors_1std[0]:.7f}")
        print(f"\delta: {delta_median:.7f} +/- {param_errors_1std[1]:.7f}")

    #Plot
    #plt.figure(figsize=(18, 14))
    plt.figure(figsize=(14, 14))
    plt.ylim(-7.5, 0.5)
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, (fradius * 0.5) + 1, step=1))
    #plt.xticks(np.arange(0, 11, step=1)) 
    plt.xlabel('$r_p/R_v$', fontsize=40)
    plt.ylabel(r'$\gamma_t(r_p) \, [10^{-3}]$', fontsize=40)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    
    if binsize==0:
        # Plot data
        if cosmoin=='LCDM':
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$ $\Lambda$CDM', color='k', fmt='o')
        elif cosmoin=='LCDM_0.15':
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$ $\Lambda$CDM$_{0.15 \, eV}$', color='k', fmt='o')
        else:
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$'fr' {cosmoin}', color='k', fmt='o')
        
    else:
        bin_edges = [1, 3.79, 4.85, 15]
        min_radius = bin_edges[binsize-1]
        max_radius = bin_edges[binsize]
        labelbin = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'

        # Plot data
        if cosmoin=='LCDM':
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$ $\Lambda$CDM'fr'{labelbin}', color='k', fmt='o')
        elif cosmoin=='LCDM_0.15':
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$ $\Lambda$CDM$_{0.15 \, eV}$'fr'{labelbin}', color='k', fmt='o')
        else:
            plt.errorbar(x_data, mean_shear * 1e3, yerr=mean_shear_error * 1e3, label=r'$\overline{\gamma_t}(r_p)$'fr'{cosmoin} {labelbin}', color='k', fmt='o')

    # Plot model best-fitt
    plt.plot(rppp, y_data_median * 1e3, label='$\gamma_t(r_p)$ model best-fit profile', color='b')

    # Plot confidence intervals:
    # 1 sigma
    plt.fill_between(x_data, lower_bounds_1std * 1e3, upper_bounds_1std * 1e3, color='b', alpha=0.3, label='Model confidence 1$\sigma$')

    # 2 sigma
    ###plt.fill_between(x_data, lower_bounds_2std * 1e3, upper_bounds_2std * 1e3, color='g', alpha=0.3, label='Model confidence 2$\sigma$')

    # 3 sigma
    ###plt.fill_between(x_data, lower_bounds_3std * 1e3, upper_bounds_3std * 1e3, color='r', alpha=0.3, label='Model confidence 3$\sigma$')

    plt.legend(fontsize=40)

def cornerplot_cosmologies(cosmoin, cobaya_path, n_cores, color, binsize=0, sigmas=None):
    import Finder_functions as mystery
    if cobaya_path == '/home/leonardo/Desktop/DUSTGRAIN-pathfinder/New_Maggiore_profile/':
        cobaya_chains_name = "cobaya_N_M_Profile"
        names = ['a', 'b', 'c', 'd', 'e']
        n_cols = (2, 3, 4, 5, 6)
    cosmolabels = []

    for cosmo in cosmoin:
        if cosmo == 'LCDM':
            cosmolabels.append(r'$\Lambda$CDM')
        elif cosmo == 'LCDM_0.15':
            cosmolabels.append(r'$\Lambda$CDM$_{0.15 \, eV}$')
        elif cosmo == 'fR4':
            cosmolabels.append(r'$f$R$4$')
        elif cosmo == 'fR5':
            cosmolabels.append(r'$f$R$5$')
        elif cosmo == 'fR6':
            cosmolabels.append(r'$f$R$6$')
            
    all_samples = []
    best_fit = None
    
    for i in reversed(range(len(cosmoin))):
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin[i]}/chains_H_diagnostic/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin[i]}/chains_bin_{binsize}_H/'
            
        MCMC = mystery.extract_cobaya(cobaya_chains_name, cobaya_dir, names, n_cores, n_cols)
        #print (np.cov(MCMC.T))
        current_best_fit = np.median(MCMC, axis=0)
    
        if i==0:
            best_fit = current_best_fit  # Save best_fit of $\Lambda$CDM
        
        if binsize==0:
            all_samples.append(mystery.prepare_sample(MCMC, f'{cosmolabels[i]}', names, names))
        else:
            bin_edges = [1, 3.79, 4.85, 15]
            min_radius = bin_edges[binsize-1]
            max_radius = bin_edges[binsize]
            labelbin = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'

            all_samples.append(mystery.prepare_sample(MCMC, f'{cosmolabels[i]}', names, names))# 'f'{labelbin}
            
    # Set general font sizes
    plt.rcParams['font.size'] = 14  # General font size
    plt.rcParams['axes.labelsize'] = 14  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 12  # X-tick label font size
    plt.rcParams['ytick.labelsize'] = 12  # Y-tick label font size
    #plt.rcParams['legend.fontsize'] = 18  # Legend font size

    m, contour_args = mystery.prepare_plot(sigmas=sigmas)

    fig = m.triangle_plot(
        all_samples, 
        filled=True, 
        label_order=-1, 
        contour_ls=["-", "-", "-", "-", "--"], 
        markers=None,
        contour_args={'alpha': 0.75, **contour_args},
        contour_colors=color[::-1])
    
    fig = plt.gcf()
    
    #if binsize == 1:
        # Parametro 'a'
        #m.subplots[0, 0].set_xlim([0.0072, 0.0085])  
        #m.subplots[1, 0].set_ylim([0.54, 0.5625])      

        # Parametro 'b'
        #m.subplots[1, 1].set_xlim([0.54, 0.5625])    
        #m.subplots[2, 1].set_ylim([2.35, 2.615])      

        # Parametro 'c'
        #m.subplots[2, 2].set_xlim([2.35, 2.615])      
        #m.subplots[3, 2].set_ylim([0.33, 0.41])     

        # Parametro 'd'
        #m.subplots[3, 3].set_xlim([0.33, 0.41])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        #m.subplots[4, 4].set_xlim([2.025, 2.25])

    #elif binsize == 2:
        # Parametro 'a'
        #m.subplots[0, 0].set_xlim([0.015, 0.02125])  
        #m.subplots[1, 0].set_ylim([0.495, 0.5225])      

        # Parametro 'b'
        #m.subplots[1, 1].set_xlim([0.495, 0.5225])    
        #m.subplots[2, 1].set_ylim([2.505, 2.72])      

        # Parametro 'c'
        #m.subplots[2, 2].set_xlim([2.505, 2.72])      
        #m.subplots[3, 2].set_ylim([-0.265, -0.075])     

        # Parametro 'd'
        #m.subplots[3, 3].set_xlim([-0.265, -0.075])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        #m.subplots[4, 4].set_xlim([1.89, 2.175])

    #elif binsize == 3:
        # Parametro 'a'
        #m.subplots[0, 0].set_xlim([0.034, 0.0505])  
        #m.subplots[1, 0].set_ylim([0.19, 0.26])      

        # Parametro 'b'
        #m.subplots[1, 1].set_xlim([0.19, 0.26])    
        #m.subplots[2, 1].set_ylim([2.265, 2.41])      

        # Parametro 'c'
        #m.subplots[2, 2].set_xlim([2.265, 2.41])      
        #m.subplots[3, 2].set_ylim([-0.83, -0.65])     

        # Parametro 'd'
        #m.subplots[3, 3].set_xlim([-0.83, -0.65])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        #m.subplots[4, 4].set_xlim([1.175, 1.625])


    # Adjust the font size for the legend and labels in the corner plot
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(22.5)  # Adjust tick label font size
        ax.xaxis.label.set_size(30)  # Adjust x-axis label font size
        ax.yaxis.label.set_size(30)  # Adjust y-axis label font size
        ax.xaxis.labelpad = 25  # Move x-axis label slightly down

    # Create a custom legend with adjusted box size
    #plt.legend(cosmoin, loc='upper right', fontsize=18, bbox_to_anchor=(1.2, 1.2))

    # Adjust the font size for the legend generated by m.triangle_plot()
    leg = fig.legends[0]  # Get the first legend
    for text in leg.get_texts():
        text.set_fontsize(30)  # Adjust legend font size

    # Move the legend slightly to the right
    leg.set_bbox_to_anchor((0.75, 0.9))  # Adjust the values as needed
    
    if binsize==0:
        plt.savefig('/home/leonardo/Desktop/pdfs2/contours.pdf', format='pdf', bbox_inches='tight')
        #plt.savefig(f'contours_cosmologies_binsize_{binsize}.pdf', format='pdf')
    elif binsize==1:
        plt.savefig('/home/leonardo/Desktop/pdfs2/piccoli.pdf', format='pdf', bbox_inches='tight')
    elif binsize==2:   
        plt.savefig('/home/leonardo/Desktop/pdfs2/medi.pdf', format='pdf', bbox_inches='tight')
    elif binsize==3:
        plt.savefig('/home/leonardo/Desktop/pdfs2/grandi.pdf', format='pdf', bbox_inches='tight')
        
def cornerplot_bins(cosmoc, cobaya_path, dirs, n_cores, color_dir, sigmas=None):
    import Finder_functions as mystery
    if cobaya_path == '/home/leonardo/Desktop/DUSTGRAIN-pathfinder/New_Maggiore_profile/':
        cobaya_chains_name = "cobaya_N_M_Profile"
        names = ['a', 'b', 'c', 'd', 'e']
        n_cols = (2, 3, 4, 5, 6)
    bin_edges = [1, 3.79, 4.85, 15]
    labels = []
    for i in range(len(dirs)):
        min_radius = bin_edges[i]
        max_radius = bin_edges[i+1]
        labell = '$R_v$ in ' + '[' + str(min_radius) + '-' + str(max_radius) + '] arcmin'
        labels.append(labell)
    cosmolabel = []    
    if cosmoc == 'LCDM':
        cosmolabel = r'$\Lambda$CDM'
    elif cosmoc == 'LCDM_0.15':
        cosmolabel = r'$\Lambda$CDM$_{0.15 \, eV}$'
    elif cosmoc == 'fR4':
        cosmolabel = r'$f$R$4$'
    elif cosmoc == 'fR5':
        cosmolabel = r'$f$R$5$'
    elif cosmoc == 'fR6':
        cosmolabel = r'$f$R$6$'

    all_samples = []
    best_fit = None  

    for i in range(len(dirs)):
    
        cobaya_dir = cobaya_path + f'{cosmoc}/chains_bin_{i+1}_H/'
        MCMC = mystery.extract_cobaya(cobaya_chains_name, cobaya_dir, names, n_cores, n_cols)
        #print (np.cov(MCMC.T))
        current_best_fit = np.median(MCMC, axis=0)
        print(f'Best Fit for {cosmoc} {i+1}:', current_best_fit)
        if i==1:
            best_fit = current_best_fit  # Save best_fit of medium voids
            
        all_samples.append(mystery.prepare_sample(MCMC, f'{cosmolabel} {labels[i]}', names, names))
    
    # Set general font sizes
    plt.rcParams['font.size'] = 14  # General font size
    plt.rcParams['axes.labelsize'] = 14  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 12  # X-tick label font size
    plt.rcParams['ytick.labelsize'] = 12  # Y-tick label font size
    #plt.rcParams['legend.fontsize'] = 18  # Legend font size

    #m = mystery.prepare_plot()
    #fig = m.triangle_plot(
        #all_samples, 
        #filled=True, 
        #markers=None,
        #contour_args={'alpha': 0.75},
        #contour_colors=color_dir)
        
    m, contour_args = mystery.prepare_plot(sigmas=sigmas) #3, 5])
    
    fig = m.triangle_plot(
        all_samples, 
        filled=True, 
        markers=None,
        contour_args={'alpha': 0.75, **contour_args},  
        contour_colors=color_dir)
    
    ##fig = m.triangle_plot(
        ##all_samples, 
        ##filled=True, 
        ##markers=None,
        ##contour_args={'alpha': 0.75, **contour_args}, 
        ##contour_colors=color_dir)#contour_ls=["-", "-", "--"],

    fig = plt.gcf()

    if cosmoc == 'LCDM':
        #''' solo per zoom per parametro a vuoti piccoli'''
        #m.subplots[0, 0].set_xlim([0.005, 0.01])  
        #m.subplots[1, 0].set_ylim([0.4, 0.6])  

        # Imposta i limiti specifici per ogni parametro
        # Parametro 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0425])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])

        # Parametro 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.28, 2.65])      

        # Parametro 'c'
        m.subplots[2, 2].set_xlim([2.28, 2.65])      
        m.subplots[3, 2].set_ylim([-0.8, 0.45])     

        # Parametro 'd'
        m.subplots[3, 3].set_xlim([-0.8, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        m.subplots[4, 4].set_xlim([1.38, 2.23])
        
    elif cosmoc == 'fR4':
        # Parametro 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0525])  
        m.subplots[1, 0].set_ylim([0.17, 0.58])      

        # Parametro 'b'
        m.subplots[1, 1].set_xlim([0.17, 0.58])    
        m.subplots[2, 1].set_ylim([2.3, 2.725])      

        # Parametro 'c'
        m.subplots[2, 2].set_xlim([2.3, 2.725])      
        m.subplots[3, 2].set_ylim([-0.9, 0.45])     

        # Parametro 'd'
        m.subplots[3, 3].set_xlim([-0.9, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        m.subplots[4, 4].set_xlim([1.15, 2.15])

    elif cosmoc == 'fR5':
        # Parametro 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0475])  
        m.subplots[1, 0].set_ylim([0.175, 0.58])      

        # Parametro 'b'
        m.subplots[1, 1].set_xlim([0.175, 0.58])    
        m.subplots[2, 1].set_ylim([2.275, 2.75])      

        # Parametro 'c'
        m.subplots[2, 2].set_xlim([2.275, 2.75])      
        m.subplots[3, 2].set_ylim([-0.85, 0.45])     

        # Parametro 'd'
        m.subplots[3, 3].set_xlim([-0.85, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        m.subplots[4, 4].set_xlim([1.23, 2.225])

        
    elif cosmoc == 'fR6':
        # Parametro 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0435])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])      

        # Parametro 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.255, 2.675])      

        # Parametro 'c'
        m.subplots[2, 2].set_xlim([2.255, 2.675])      
        m.subplots[3, 2].set_ylim([-0.8, 0.45])     

        # Parametro 'd'
        m.subplots[3, 3].set_xlim([-0.8, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        m.subplots[4, 4].set_xlim([1.35, 2.23])

        
    elif cosmoc == 'LCDM_0.15':    
        # Parametro 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.041])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])      

        # Parametro 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.25, 2.6])      

        # Parametro 'c'
        m.subplots[2, 2].set_xlim([2.25, 2.6])      
        m.subplots[3, 2].set_ylim([-0.75, 0.45])     

        # Parametro 'd'
        m.subplots[3, 3].set_xlim([-0.75, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parametro 'e'
        m.subplots[4, 4].set_xlim([1.385, 2.265])
    

    # Adjust the font size for the legend and labels in the corner plot
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(25)  # Adjust tick label font size
        ax.xaxis.label.set_size(32.5)  # Adjust x-axis label font size
        ax.yaxis.label.set_size(32.5)  # Adjust y-axis label font size
        ax.xaxis.labelpad = 25  # Move x-axis label slightly down
        

    # Create a custom legend with adjusted box size
    #plt.legend(cosmoin, loc='upper right', fontsize=18, bbox_to_anchor=(1.2, 1.2))

    # Adjust the font size for the legend generated by m.triangle_plot()
    leg = fig.legends[0]  # Get the first legend
    for text in leg.get_texts():
        text.set_fontsize(30)  # Adjust legend font size

    # Move the legend slightly to the right
    leg.set_bbox_to_anchor((0.775, 0.9))  # Adjust the values as needed

    plt.savefig('/home/leonardo/Desktop/pdfs2/total_contours_3bins.pdf', format='pdf', bbox_inches='tight')

    plt.savefig('/home/leonardo/Desktop/pdfs/total_contours_3bins.pdf', format='pdf', bbox_inches='tight')
