#import daft
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import astropy.cosmology as cosmology
import pandas as pd
import numpy as np
import bisect
import scipy.stats as sps
import scipy.interpolate as spi
import scipy.optimize as spo
import normalise_funs as nf
import fit_funs as ff
import pickle
import probgen as pg
import pylab as pl
z_sigma = 0.03
#import hickle
name_save ='newconf_newcosmo_newdist'
import os
paths = ['data', 'plots']
if not os.path.exists('data'):
    print('WARNING: You will need to put some data files in the `data` directory to have nontrivial mock data.')
        
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

import sys
log_epsilon = sys.float_info.min_exp
epsilon = sys.float_info.min

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)


types = ['Ia', 'Ibc', 'II']
colors = ['b', 'm', 'g']
n_types = len(types)

# making up the type fractions, will replace this with data soon!
frac_types = np.array([0.4, 0.2, 0.4]) # removing this from most places and making the types redshift dependent

# these arbitrary limits are from the selection function
min_z = 0.05
max_z = 0.6


n_of_z_consts = {}
n_of_z_consts['Ia'] = (1.5, 1.0)
n_of_z_consts['Ibc'] = (1., 0.8)
n_of_z_consts['II'] = (0.5, 0.5)

true_n_of_z = []
for t in types:
    (mean, std) = n_of_z_consts[t]
    low, high = (min_z - mean) / std, (max_z - mean) / std
    print(low,high)
    true_n_of_z.append(sps.truncnorm(low, high, loc = mean, scale = std))

plot_res = 20
z_range = max_z - min_z
z_grid = np.linspace(min_z, max_z, num=plot_res + 1, endpoint=True) # making sure edges are wide enough
z_plot = (z_grid[1:] + z_grid[:-1]) / 2.
z_dif_plot = z_grid[1:] - z_grid[:-1]


def inverter(z, mu, H0, Om0, Ode0, hyperparams):
    import astropy.cosmology as cosmology
    #note: this inverter is slow! perhaps we could speed it up with interpolation over predefined grids?
    def cosmo_helper(hyperparams):
        return np.array([abs(cosmology.w0waCDM(H0, Om0, Ode0, w0=hyperparams[0], wa=hyperparams[1]).distmod(z).value - mu)])
    solved_cosmo = spo.minimize(cosmo_helper, hyperparams, method="Nelder-Mead", options={"maxfev": 1e5, "maxiter":1e5})
    prob = interim_dist.pdf(solved_cosmo.x)
    return prob


def safe_log(arr, threshold=epsilon):
    arr[arr < threshold] = threshold
    return np.log(arr)

def reg_vals(arr, threshold=log_epsilon):
    arr[arr < threshold] = threshold
    return arr

                

plot_true_n_of_z = np.zeros((n_types, plot_res))


for t in range(n_types):
    plot_true_n_of_z[t] = true_n_of_z[t].pdf(z_plot)
    plot_true_n_of_z[t] = frac_types[t, np.newaxis] * np.array(plot_true_n_of_z)[t]

for z in range(len(z_dif_plot)):
    plot_true_n_of_z[:,z] /= np.sum(plot_true_n_of_z[:,z])



for t in range(n_types):
    plt.plot(z_plot, plot_true_n_of_z[t], color=colors[t], label=types[t])
plt.xlabel('z')
plt.ylabel('relative rate')
#plt.legend()
#plt.savefig('plots/true_rates.png')



def sample_discrete(fracs, n_of_z, N):
# RH needs to focus on this part
    found_types = [0, 0, 0]
    poster_indices = []
    out_info = []
    cdf = np.cumsum(fracs)
    for n in range(N):
        each = {}
        r = np.random.random()
        k = bisect.bisect(cdf, r)
        each['t'] = types[k]
        each['z'] = n_of_z[k].rvs()
        out_info.append(each)
    return out_info

n_sne = 50
true_id = range(n_sne)

true_params = sample_discrete(frac_types, true_n_of_z, n_sne)
true_zs = [true_param['z'] for true_param in true_params]
true_types = [true_param['t'] for true_param in true_params]
posters = []
posters.append(next(i for i,v in enumerate(true_types) if v == 'Ia'))
posters.append(next(i for i,v in enumerate(true_types) if v == 'Ibc'))
posters.append(next(i for i,v in enumerate(true_types) if v == 'II'))

to_plot = [[d['z'] for d in true_params if d['t'] == types[t]] for t in range(n_types)]
hist_bins = np.linspace(min_z, max_z, plot_res + 1)
bin_difs = hist_bins[1:] - hist_bins[:-1]

#for t in range(n_types):
#    plt.plot(z_plot, plot_true_n_of_z[t] * n_sne * bin_difs, color=colors[t], label='true '+types[t])
#    plt.hist(to_plot[t], bins=hist_bins, color=colors[t], alpha=1./3., label='sampled '+types[t], normed=False)

#plt.xlabel(r'$z$')

#plt.ylabel(r'relative rate')
#plt.legend(fontsize='xx-small')
#plt.savefig('plots/obs_rates.png')

# Planck 2015 results XIV. Dark energy and modified gravity - Figure 3
true_H0 = 67.9
true_Ode0 = 0.693
true_Om0 = 1. - true_Ode0
true_w0 = -1.0
true_wa = 0.00
true_hyperparams = np.array([true_w0, true_wa])
n_hyperparams = len(true_hyperparams)

true_model = np.array([true_H0,true_Ode0, true_Om0, true_w0, true_wa])
vary_model = np.array([0,0,0,1,1])
#true_cosmo = cosmology.FlatLambdaCDM(H0=true_H0, Om0=true_Om0)
true_cosmo = cosmology.w0waCDM(true_H0, true_Om0, true_Ode0, w0=true_w0, wa=true_wa)

for n in range(n_sne):
        true_params[n]['mu'] = true_cosmo.distmod(true_params[n]['z']).value


        # this binning is arbitrary!
n_zs = 101
z_bins = np.linspace(min_z*(1.05), 0.95*max_z, num=n_zs, endpoint=True)
#making sure the simulated redshifts always within the true!
z_difs = z_bins[1:] - z_bins[:-1]
z_dif = np.mean(z_difs)
z_mids = (z_bins[1:] + z_bins[:-1]) / 2.

mu_lims = (true_cosmo.distmod(min_z).value, true_cosmo.distmod(max_z).value)

# want this to be agnostic about true cosmology
n_mus = 101
(min_mu, max_mu) = mu_lims
#mu_lims[0] - np.random.random(), mu_lims[1] + np.random.random()#min([s['mu'] for s in true_params]) - 0.5, max([s['mu'] for s in true_params]) + 0.5
mu_bins = np.linspace(min_mu, max_mu, num=n_mus, endpoint=True)
mu_difs = mu_bins[1:] - mu_bins[:-1]
mu_dif = np.mean(mu_difs)
mu_range = np.max(mu_bins) - np.min(mu_bins)
#print(mu_bins)
mu_mids = (mu_bins[1:] + mu_bins[:-1]) / 2.

z_mu_grid = np.array([[(z, mu) for mu in mu_mids] for z in z_mids])
cake_shape = np.shape(z_mu_grid)
unity = np.ones((n_sne, n_types, n_zs-1, n_mus-1))
unity_t = np.ones(n_types)
unity_z = np.ones(n_zs-1)
unity_all = np.ones((n_sne, n_types, n_zs-1, n_mus-1))
unity_hubble = np.ones((n_zs-1, n_mus-1))
unity_zt = np.ones((n_types, n_zs-1))
unity_one = np.ones((n_types, n_zs-1, n_mus-1))  
 
pmin, pmax = log_epsilon, np.log(1./(min(z_difs) * min(mu_difs)))

binned_n_of_z = np.zeros((n_types, n_zs-1))

rate_of_z = np.zeros((n_types, n_zs-1))
for t in range(n_types):
    rate_of_z[t] = frac_types[t, np.newaxis]*true_n_of_z[t].pdf(z_mids)
    cdfs = true_n_of_z[t].cdf(z_bins)
    binned_n_of_z[t] = (cdfs[1:] - cdfs[:-1])
    binned_n_of_z = frac_types[t, np.newaxis]*np.array(binned_n_of_z) #

rate_of_z = np.array(rate_of_z)#

# Making an N(z) vector from which to generate the confusion matrix
for z in range(len(z_difs)):
    rate_of_z[:,z]/=np.sum(rate_of_z[:,z])

#print(np.sum(rate_of_z[:,:],axis=0), 'sum over type')
#print(np.sum(rate_of_z[:,:],axis=1), 'sum over z')

pl.clf()
for t in range(n_types):
    pl.plot(z_mids[:], rate_of_z[t,:], color=colors[t], label=types[t])
    
pl.plot(z_mids, np.sum(rate_of_z[:,:],axis=0), 'k')
pl.xlabel('z')
pl.ylabel('relative rate')
#pl.savefig('test_prosb.png')
conf_matrix = np.zeros((len(z_difs),n_types,n_types))

# Generating new probabilities
for i in range(len(z_difs)):
    for t in range(n_types):
        for tt in range(n_types):
            conf_matrix[i][t][tt] = 0.25*np.sqrt(rate_of_z[tt,i]*rate_of_z[t,i]) # taking small off diagonal
        conf_matrix[i][t][t] = rate_of_z[t,i]



print(conf_matrix[0:3][:][:])
#np.random.rand(3,3) + (0.8 * np.eye(3))
# RH removed this because we don't think that classifiation probabilities
# need to know about sn type* frac_types[:, np.newaxis]
print('---')
# we want to sum over the axis that is the "true type"  axis -- which we will see later, is the column.
# this is because over all light curve fitters, we want the probability to sum to unity, but we don't expect the
# objects to sum to unity across one fitter

#norm_prob = np.sum(conf_matrix, axis=1)
#print(norm_prob)
#print('norm prob above')
#conf_matrix[0] /= norm_prob[0]
#conf_matrix[1] /= norm_prob[1]
#conf_matrix[2] /= norm_prob[2]

#print(conf_matrix)
#print('---')

#true_rates = np.sum(conf_matrix, axis=1)
#obs_rates = np.sum(conf_matrix, axis=0)
#print(conf_matrix, frac_types, true_rates, obs_rates)


# Renee removed this normalisation for now - as I don't understand it!

#conf_matrix /= true_rates[:, np.newaxis]
#assert np.all(np.isclose(true_rates, frac_types))


# print(conf_matrix)



ln_conf_matrix = safe_log(conf_matrix)

Ia_Ia_var = np.array([0.001, 0.04]) ** 2
Ibc_Ia_delta = 0.1 #np.mean(mu_mids)
Ibc_Ia_var = np.array([0.001, 0.04]) ** 2
II_Ia_delta = 0.1 # np.mean(mu_mids)
II_Ia_var = np.array([0.001, 0.04]) ** 2



def fit_Ia(z, mu, vb=False):

    #     cake = unity_hubble#np.zeros((n_types, n_zs-1, n_mus-1))
    cake_Ia = sps.multivariate_normal(mean = np.array([z, mu]), cov = Ia_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_Ia.rvs()
    cake_Ia = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = Ia_Ia_var * np.eye(2))
    cake = nf.normalize_hubble(cake_Ia.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), z_difs, mu_difs, vb=vb)
    return cake

def fit_Ibc(z, mu, vb=False):
    
    
    #     cake = np.zeros((n_types, n_zs-1, n_mus-1))
    cake_Ibc = sps.multivariate_normal(mean = np.array([z, mu - Ibc_Ia_delta]), cov = Ibc_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_Ibc.rvs()

    cake_Ibc = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = Ibc_Ia_var * np.eye(2))
    cake = nf.normalize_hubble(cake_Ibc.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), z_difs, mu_difs, vb=vb)
    return cake



def fit_II(z, mu, vb=False):

    #     cake = np.zeros((n_types, n_zs-1, n_mus-1))
    cake_II = sps.multivariate_normal(mean = np.array([z, mu - Ibc_Ia_delta]), cov = II_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_II.rvs()
    cake_II = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = II_Ia_var * np.eye(2))
    cake = nf.normalize_hubble(cake_II.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), z_difs, mu_difs, vb=vb)
    return cake


def fit_any(true_vals, vb=False):

    ind = np.where(true_vals['z']<z_mids)[0]
#    print(ind, 'testing empty')

    if len(ind) == 0: 
        ind=0
    else:
        ind = np.min(ind)

 #   print(true_vals['z'], 'z val')
 #   print(ind, 'indices')
 #   print(z_mids[ind], 'zvals')
    #     print(unity_one)
    cake = np.zeros((n_types, n_zs-1, n_mus-1))#unity_one.copy()
    #     print(unity_one)
    if true_vals['t'] == 'Ia':
#        print('here in Ia')
        cake[0] = fit_Ia(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[ind][0][:]
#        print(ln_conf,true_vals['t'])
    if true_vals['t'] == 'Ibc':
 #       print('here in Ibc')
        cake[0] = fit_Ibc(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[ind][1][:]
 #       print(ln_conf,true_vals['t'])
    if true_vals['t'] == 'II':
  #      print('here in II')
        cake[0] = fit_II(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[ind][2][:]
  #      print(ln_conf, true_vals['t'])

    if vb: print(np.exp(ln_conf))
    dist = sps.norm(loc = true_vals['z'], scale = z_sigma)
    z_means = dist.rvs(2)
    layer_Ibc = sps.norm(loc = z_means[0], scale = z_sigma).pdf(z_mids)
    #     print(layer_Ibc)
    layer_II = sps.norm(loc = z_means[1], scale = z_sigma).pdf(z_mids)
    #     print(layer_II)
    cake[1] = nf.normalize_hubble(unity_hubble * layer_Ibc[:, np.newaxis], z_difs, mu_difs, vb=vb)
    cake[2] = nf.normalize_hubble(unity_hubble * layer_II[:, np.newaxis], z_difs, mu_difs, vb=vb)
    #     cake = normalize_one(cake)
    if not np.all(cake>=0.):
        print(true_vals)
        assert False
#    print(np.shape(cake), np.shape(ln_conf))
    cake = reg_vals(safe_log(cake) + ln_conf[:, np.newaxis, np.newaxis])
    #     cake = safe_log(normalize_one(np.exp(cake)))
    if vb: print(np.sum(np.sum(np.exp(cake) * z_difs[np.newaxis, :, np.newaxis] * mu_difs[np.newaxis, np.newaxis, :], axis=2), axis=1))
    return cake


def fit_all(catalog, vb=False):
    dessert = []
    i=0
    for true_vals in catalog:
        if vb: print(i)
        thing = fit_any(true_vals, vb=vb)
        try:
            dessert.append(thing)
        except AssertionError:
            print('error '+str(thing))
            i += 1
    return np.array(dessert)

sheet_cake = fit_all(true_params, vb=False)
sheet_cake = reg_vals(safe_log(nf.normalize_all(np.exp(sheet_cake), z_difs, mu_difs, vb=False)))



if not (os.path.exists('data/ratios_wfd.txt') and os.path.exists('data/ratios_ddf.txt')):
    print('WARNING: No SN LC selection function data found in `data` directory, using flat SN LC selection function instead.')
    sn_sel_fun_in = unity_zt.copy() # RH changed this
    sn_sel_fun_in=sn_sel_fun_in[0:len(z_bins)]
    #print(sn_sel_fun_in,len(z_bins))
else:
    with open('data/ratios_wfd.txt', 'r') as wfd_file:
        #     wfd_file.next()
        tuples = (line.split(None) for line in wfd_file)
        wfddata = [[pair[k] for k in range(0,len(pair))] for pair in tuples]
        n_sel_fun_zs = 6
        zs_eval = np.array([float(wfddata[i][0]) for i in range(1, n_sel_fun_zs)])
        wfd_data = np.array([np.array([int(wfddata[i][2 * j]) for j in range(1, (len(wfddata[i]))/2+1)]) for i in range(1, n_sel_fun_zs)])
        wfd_data[np.isnan(wfd_data)] = 0.
        # print(wfd_data)
        with open('data/ratios_ddf.txt', 'r') as ddf_file:
            #     ddf_file.next()
            tuples = (line.split(None) for line in ddf_file)
            ddfdata = [[pair[k] for k in range(0,len(pair))] for pair in tuples]
            ddf_data = np.array([np.array([float(ddfdata[i][2 * j]) for j in range(1, (len(ddfdata[i]))/2+1)]) for i in range(1, n_sel_fun_zs)])
            # print(ddf_data)
            # these are the recovery rates
            sn_sel_fun_in = np.transpose(wfd_data / ddf_data)
            # sn_sel_fun = sn_sel_fun.T
            # print(sn_sel_fun)
            # It's actually a big problem for the selection function to go to 0 or exceed 1.
            # Note: these need to be normalized by survey volume so are not valid at this time!

# #need this to be # types * # z bins in shape
sn_sel_fun_out = np.ones((n_types, n_zs-1))
sn_sel_fun_z = nf.normalize_zt(sn_sel_fun_out, z_difs, vb=True)


sn_sel_fun = nf.normalize_one(sn_sel_fun_z[:, :, np.newaxis] * unity_one, z_difs, mu_difs, vb=True)
ln_sn_selection_function = safe_log(sn_sel_fun)
selfunmin, selfunmax = np.min(ln_sn_selection_function), np.max(ln_sn_selection_function)
sn_interim_nz = np.ones((n_types, n_zs-1))#this is flat, replace this with nontrivial interim prior on types and redshifts
sn_interim_nz = nf.normalize_zt(sn_interim_nz, z_difs)
sn_interim_nz = nf.normalize_one(unity_one * sn_interim_nz[:, :, np.newaxis], z_difs, mu_difs)
ln_sn_interim_nz = safe_log(sn_interim_nz)



# read these off Planck plots
interim_H0 = true_H0#70.0
delta_H0 = 2.2# * 10.
interim_Om0 = true_Om0#1. - 0.721
delta_Om0 = 0.025# * 10.
interim_Ode0 = true_Ode0
delta_Ode0 = 0.025 * 10.
interim_w0 = -1.0#true_w0
delta_w0 = 0.05#0.2# * 10.
interim_wa = 0.0#true_wa
delta_wa = 0.05#1.5# * 10.
interim_cosmo_hyperparams = np.array([interim_w0, interim_wa])
interim_cosmo_hyperparam_sigmas = np.array([delta_w0, delta_wa])
interim_cosmo_hyperparam_vars = (interim_cosmo_hyperparam_sigmas) * np.eye(n_hyperparams)
interim_dist = sps.multivariate_normal(mean = interim_cosmo_hyperparams, cov = interim_cosmo_hyperparam_vars)
interim_cosmo = cosmology.w0waCDM(interim_H0, interim_Om0, interim_Ode0, w0=interim_w0, wa=interim_wa)

sn_interim_hubble = np.zeros((n_zs-1, n_mus-1))
for z in range(n_zs-1):
        for mu in range(n_mus-1):
            prob = inverter(z_mids[z], mu_mids[mu], H0=interim_H0, Om0=interim_Om0, Ode0=interim_Ode0, hyperparams = interim_cosmo_hyperparams)
            sn_interim_hubble[z][mu] = prob
sn_interim_hubble = nf.normalize_hubble(sn_interim_hubble, z_difs, mu_difs)
ln_sn_interim_hubble = safe_log(sn_interim_hubble)
sn_interim_hubble = nf.normalize_one(unity_one * sn_interim_hubble[np.newaxis, :, :], z_difs, mu_difs)
ln_sn_interim_hubble = safe_log(sn_interim_hubble)
ln_sn_interim = reg_vals(ln_sn_interim_nz + ln_sn_interim_hubble)
sn_interim = np.exp(ln_sn_interim)
ln_sn_interim = safe_log(nf.normalize_one(sn_interim, z_difs, mu_difs))
            
           
# very simple p(z) model, simple gaussians, but binned parametrization permits arbitrary shapes
pz_sigma = 0.03

pzs, ln_pzs = [], []
for s in range(n_sne):
    dist = sps.norm(loc = true_params[s]['z'], scale = pz_sigma)
    pz_mean = dist.rvs()
    new_dist = sps.norm(loc = pz_mean, scale = pz_sigma)
    pz = new_dist.pdf(z_mids)
    #ln_pz = new_dist.logpdf(z_mids)
    pzs.append(pz)
    #ln_pzs.append(ln_pz)
pzs = np.array(pzs)
pzs = nf.normalize_z(pzs, z_difs)
ln_pzs = safe_log(pzs)#np.array(ln_pzs)

# We emulate this using data from a realistic galaxy simulation.
# We want the number of galaxies as a function of redshift, SED type, and luminosity.
# (Buzzard, for example, includes this.)
# Using a realistic set of magnitude limits, we want to calculate the recovered fraction
# of galaxies as a function of redshift, SED type, and luminosity.
# Then we just integrate over SED type and luminosity to get $p(z | \vec{\beta})$

# pz_selfun = np.ones(n_zs-1)
# pz_selfun /= np.sum(pz_selfun * z_difs)
# assert np.isclose(np.sum(pz_selfun * z_difs), 1.)
# ln_pz_selfun = safe_log(pz_selfun)

host_sel_fun = np.ones(n_zs-1)
host_sel_fun = nf.normalize_z(host_sel_fun, z_difs)
# host_sel_fun_norm = np.sum(host_sel_fun * z_difs)
# host_sel_fun /= host_sel_fun_norm
# assert np.isclose(np.sum(host_sel_fun * z_difs), 1.)
ln_host_selection_function = safe_log(host_sel_fun)


# read in the SDSS DR7 one instead
# separate interim prior from LC fitter and photo-z PDFs: this is for photo-z PDFs, flat for now, replace with SDSS n(z)
pz_interim = np.ones(n_zs-1)
rim = nf.normalize_z(pz_interim, z_difs)
# pz_interim /= np.sum(pz_interim * z_difs)
# assert np.isclose(np.sum(pz_interim * z_difs), 1.)
ln_pz_interim = safe_log(pz_interim)


# Combining everything
ln_host_probs = reg_vals(ln_pzs + ln_host_selection_function[np.newaxis, :] + ln_pz_interim[np.newaxis, :])
host_probs = np.exp(ln_host_probs)[:, np.newaxis, :, np.newaxis] * unity_all
host_probs = nf.normalize_all(host_probs, z_difs, mu_difs)
ln_host_probs = safe_log(host_probs)
ln_sn_probs = reg_vals(sheet_cake + ln_sn_selection_function[np.newaxis, :] + ln_sn_interim[np.newaxis, :])
sn_probs = np.exp(ln_sn_probs)
sn_probs = nf.normalize_all(sn_probs, z_difs, mu_difs)
ln_sn_probs = safe_log(sn_probs)
interim_ln_posteriors = reg_vals(ln_host_probs + ln_sn_probs)
interim_posts = np.exp(interim_ln_posteriors)
interim_posts = nf.normalize_all(interim_posts, z_difs, mu_difs)
interim_ln_posteriors = safe_log(interim_posts)

sn_id = ['CID_%i'%n for n in np.arange(0,n_sne,1)]


# write true hyperparameters just to check
#d = {'b' : 1, 'a' : 0, 'c' : 2}
truth = { 'phi': binned_n_of_z, 'model': true_model, 'vary_index': vary_model, 'data': true_params, 'id': sn_id}
outfile = open('data/truthfile_%s.pkl'%name_save,'wb')
pickle.dump(truth,outfile)
outfile.close()

#Saving the redshift posterior
output_z = {'types': types, 'z_bins': z_bins, 'mu_bins': mu_bins}
output_z['ln host selection function'] = ln_host_selection_function
output_z['host interim ln prior'] = ln_pz_interim
output_z['ln host posterior'] = ln_host_probs
output_z['id'] = sn_id
with open('data/hostzfile_%s.pkl'%name_save, 'w') as out_file:
    pickle.dump(output_z, out_file)

# Saving the LC posterior
output_lc = {'types': types, 'z_bins': z_bins, 'mu_bins': mu_bins}
output_lc['ln sn selection function'] = ln_sn_selection_function
output_lc['sn interim ln prior'] = ln_sn_interim
#output_lc['ln selection function'] = safe_log(normalize_one(np.exp(reg_vals(output['ln host selection function'][np.newaxis, :, np.newaxis] + output['ln sn selection function']))))
output_lc['ln sn posterior'] = ln_sn_probs
output_lc['id'] = sn_id

with open('data/snfile_%s.pkl'%name_save, 'w') as out_file:
    pickle.dump(output_lc, out_file)

#Saving the full posterior
output = {'types': types, 'z_bins': z_bins, 'mu_bins': mu_bins}
output['ln host selection function'] = ln_host_selection_function
output['host interim ln prior'] = ln_pz_interim
output['ln sn selection function'] = ln_sn_selection_function
output['sn interim ln prior'] = ln_sn_interim
output['ln selection function'] = safe_log(nf.normalize_one(np.exp(reg_vals(output['ln host selection function'][np.newaxis, :, np.newaxis] + output['ln sn selection function'])),z_difs, mu_difs))
output['interim ln prior'] = safe_log(nf.normalize_one(np.exp(reg_vals(output['host interim ln prior'][np.newaxis, :, np.newaxis] + output['sn interim ln prior'])),z_difs, mu_difs))
output['ln prior info'] = safe_log(nf.normalize_one(np.exp(reg_vals(output['interim ln prior'] + output['ln selection function'])),z_difs, mu_difs))
output['interim ln posteriors'] = interim_ln_posteriors
output['id'] = sn_id
with open('data/jointfile_%s.pkl'%name_save, 'w') as out_file:
    pickle.dump(output, out_file)

