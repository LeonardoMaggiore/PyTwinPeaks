import numpy as np
import math
import random as rnd
import warnings
import imageio
from astropy.io import fits
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
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
    
    # Generate a temporary plot to retrieve the actual contour levels
    temp_plot = plots.GetDistPlotter()
    temp_plot.settings.num_plot_contours = sigmas  # use n default contour levels from GetDist
    temp_levels = temp_plot.settings.num_plot_contours  # GetDist tells us how many it uses by default

    print(f"Actual confidence levels used by GetDist: {temp_levels}")

    contour_args = {}

    #if sigmas:
        # Retrieve the actual confidence levels
        #confidence_levels = np.linspace(0.6827, 0.9999994, len(sigmas))
        #print(f"Confidence levels standardised to match GetDist: {confidence_levels}")
        
        # Normalise to align with the GetDist method
        #max_conf = 1.0
        #confidence_levels = [level / max_conf for level in confidence_levels]
        
        #contour_args['contour_levels'] = confidence_levels
        #print(f"Normalized contour levels: {contour_args['contour_levels']}")
    
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
    for nn in range(len(rp)): #loop over all projected radii
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
    from pyTwinPeaks import Finder_functions as mystery
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
    for num in R_l: #loop over all projected radii of the grid
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

    for nn in range(len(rp)): #loop over all projected radii
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
    from pyTwinPeaks import Finder_functions as mystery
    rangeint = 1147.9660518953947 #D_z/2
    rho_m = 8.699385581016718*1e10
    Rv_peak = 1.2500443529452991
    R_l = np.linspace(1e-4, 1e1, 1000)
    delta_senza_dc, _, _ = mystery.calcola_delta_unit_stack_T(R_l, rp, ss, Rv_peak, rho_m, rangeint)
    shear_model_fit = delta_senza_dc * np.abs(delta_prime)
    return shear_model_fit.flatten()

def best_fit_model_mcmc(pre_path, cosmoin, n_cores, cobaya_path, fradius, binsize=0):
    from pyTwinPeaks import Finder_functions as mystery
    from tqdm import tqdm
    # Caricamento dati shear
    if binsize==0:
        rp, mean_shear, mean_shear_error = np.genfromtxt(pre_path + '/outputs/' + f'{cosmoin}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
    else:
        rp, mean_shear, mean_shear_error = np.genfromtxt(pre_path + '/outputs/' + f'{cosmoin}_binsize_{binsize}_mean_shear.dat', usecols=(0, 1, 2), unpack=True, skip_header=1)
        
    rp = rp[1:]
    mean_shear = mean_shear[1:]
    mean_shear_error = mean_shear_error[1:]
    
    # Use relative or user-provided cobaya_path
    if "New_Maggiore_profile" in cobaya_path:
        cobaya_chains_name = "cobaya_N_M_Profile"
        fitting_function = mystery.new_function
        names = ['a', 'b', 'c', 'd', 'e']
        n_cols = (2, 3, 4, 5, 6)
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin}/chains/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin}_binsize_{binsize}/chains/'
            
    elif "N_profile" in cobaya_path:
        cobaya_chains_name = "cobaya_N_Profile"
        fitting_function = mystery.calculate_shear_model_fit_N
        names = ['rs/R_v', '\\alpha', '\\beta', '\delta']
        n_cols = (2, 3, 4, 5)
        if binsize==0:
            cobaya_dir = cobaya_path + f'{cosmoin}/chains/'
        else:
            cobaya_dir = cobaya_path + f'{cosmoin}_binsize_{binsize}/chains/'
    
    elif "T_profile" in cobaya_path:
        cobaya_chains_name = "cobaya_T_Profile"
        fitting_function = mystery.calculate_shear_model_fit_T
        names = ['s', '\delta']
        n_cols = (2, 3)
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
    from pyTwinPeaks import Finder_functions as mystery
    if "New_Maggiore_profile" in cobaya_path:
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
        # Parameter 'a'
        #m.subplots[0, 0].set_xlim([0.0072, 0.0085])  
        #m.subplots[1, 0].set_ylim([0.54, 0.5625])      

        # Parameter 'b'
        #m.subplots[1, 1].set_xlim([0.54, 0.5625])    
        #m.subplots[2, 1].set_ylim([2.35, 2.615])      

        # Parameter 'c'
        #m.subplots[2, 2].set_xlim([2.35, 2.615])      
        #m.subplots[3, 2].set_ylim([0.33, 0.41])     

        # Parameter 'd'
        #m.subplots[3, 3].set_xlim([0.33, 0.41])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        #m.subplots[4, 4].set_xlim([2.025, 2.25])

    #elif binsize == 2:
        # Parameter 'a'
        #m.subplots[0, 0].set_xlim([0.015, 0.02125])  
        #m.subplots[1, 0].set_ylim([0.495, 0.5225])      

        # Parameter 'b'
        #m.subplots[1, 1].set_xlim([0.495, 0.5225])    
        #m.subplots[2, 1].set_ylim([2.505, 2.72])      

        # Parameter 'c'
        #m.subplots[2, 2].set_xlim([2.505, 2.72])      
        #m.subplots[3, 2].set_ylim([-0.265, -0.075])     

        # Parameter 'd'
        #m.subplots[3, 3].set_xlim([-0.265, -0.075])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        #m.subplots[4, 4].set_xlim([1.89, 2.175])

    #elif binsize == 3:
        # Parameter 'a'
        #m.subplots[0, 0].set_xlim([0.034, 0.0505])  
        #m.subplots[1, 0].set_ylim([0.19, 0.26])      

        # Parameter 'b'
        #m.subplots[1, 1].set_xlim([0.19, 0.26])    
        #m.subplots[2, 1].set_ylim([2.265, 2.41])      

        # Parameter 'c'
        #m.subplots[2, 2].set_xlim([2.265, 2.41])      
        #m.subplots[3, 2].set_ylim([-0.83, -0.65])     

        # Parameter 'd'
        #m.subplots[3, 3].set_xlim([-0.83, -0.65])    
        #m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
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
        plt.savefig(os.path.join(cobaya_dir, "contours.pdf"), format='pdf', bbox_inches='tight')
        #plt.savefig(f'contours_cosmologies_binsize_{binsize}.pdf', format='pdf')
    elif binsize==1:
        plt.savefig(os.path.join(cobaya_dir, "piccoli.pdf"), format='pdf', bbox_inches='tight')
    elif binsize==2:   
        plt.savefig(os.path.join(cobaya_dir, "medi.pdf"), format='pdf', bbox_inches='tight')
    elif binsize==3:
        plt.savefig(os.path.join(cobaya_dir, "grandi.pdf"), format='pdf', bbox_inches='tight')
        
def cornerplot_bins(cosmoc, cobaya_path, dirs, n_cores, color_dir, sigmas=None):
    from pyTwinPeaks import Finder_functions as mystery
    if "New_Maggiore_profile" in cobaya_path:
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
        #''' only for zoom on the parameter for small voids'''
        #m.subplots[0, 0].set_xlim([0.005, 0.01])  
        #m.subplots[1, 0].set_ylim([0.4, 0.6])  

        # Sets specific limits for each parameter
        # Parameter 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0425])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])

        # Parameter 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.28, 2.65])      

        # Parameter 'c'
        m.subplots[2, 2].set_xlim([2.28, 2.65])      
        m.subplots[3, 2].set_ylim([-0.8, 0.45])     

        # Parameter 'd'
        m.subplots[3, 3].set_xlim([-0.8, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        m.subplots[4, 4].set_xlim([1.38, 2.23])
        
    elif cosmoc == 'fR4':
        # Parameter 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0525])  
        m.subplots[1, 0].set_ylim([0.17, 0.58])      

        # Parameter 'b'
        m.subplots[1, 1].set_xlim([0.17, 0.58])    
        m.subplots[2, 1].set_ylim([2.3, 2.725])      

        # Parameter 'c'
        m.subplots[2, 2].set_xlim([2.3, 2.725])      
        m.subplots[3, 2].set_ylim([-0.9, 0.45])     

        # Parameter 'd'
        m.subplots[3, 3].set_xlim([-0.9, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        m.subplots[4, 4].set_xlim([1.15, 2.15])

    elif cosmoc == 'fR5':
        # Parameter 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0475])  
        m.subplots[1, 0].set_ylim([0.175, 0.58])      

        # Parameter 'b'
        m.subplots[1, 1].set_xlim([0.175, 0.58])    
        m.subplots[2, 1].set_ylim([2.275, 2.75])      

        # Parameter 'c'
        m.subplots[2, 2].set_xlim([2.275, 2.75])      
        m.subplots[3, 2].set_ylim([-0.85, 0.45])     

        # Parameter 'd'
        m.subplots[3, 3].set_xlim([-0.85, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        m.subplots[4, 4].set_xlim([1.23, 2.225])

        
    elif cosmoc == 'fR6':
        # Parameter 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.0435])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])      

        # Parameter 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.255, 2.675])      

        # Parameter 'c'
        m.subplots[2, 2].set_xlim([2.255, 2.675])      
        m.subplots[3, 2].set_ylim([-0.8, 0.45])     

        # Parameter 'd'
        m.subplots[3, 3].set_xlim([-0.8, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
        m.subplots[4, 4].set_xlim([1.35, 2.23])

        
    elif cosmoc == 'LCDM_0.15':    
        # Parameter 'a'
        m.subplots[0, 0].set_xlim([0.006, 0.041])  
        m.subplots[1, 0].set_ylim([0.205, 0.58])      

        # Parameter 'b'
        m.subplots[1, 1].set_xlim([0.205, 0.58])    
        m.subplots[2, 1].set_ylim([2.25, 2.6])      

        # Parameter 'c'
        m.subplots[2, 2].set_xlim([2.25, 2.6])      
        m.subplots[3, 2].set_ylim([-0.75, 0.45])     

        # Parameter 'd'
        m.subplots[3, 3].set_xlim([-0.75, 0.45])    
        m.subplots[4, 3].set_ylim([-0.1, 0.2])     

        # Parameter 'e'
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

    plt.savefig(os.path.join(cobaya_dir, "total_contours_3bins.pdf"), format='pdf', bbox_inches='tight')
