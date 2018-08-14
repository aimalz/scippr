import numpy as np
def fit_Ia(z, mu, vb=False):
    n_zs = len(z)
    n_mus = len(mu)
    #     cake = unity_hubble#np.zeros((n_types, n_zs-1, n_mus-1))
    cake_Ia = sps.multivariate_normal(mean = np.array([z, mu]), cov = Ia_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_Ia.rvs()
    cake_Ia = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = Ia_Ia_var * np.eye(2))
    cake = normalize_hubble(cake_Ia.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), vb=vb)
    return cake

def fit_Ibc(z, mu, vb=False):
    n_zs = len(z)
    n_mus = len(mu)

    #     cake = np.zeros((n_types, n_zs-1, n_mus-1))
    cake_Ia = sps.multivariate_normal(mean = np.array([z, mu - Ibc_Ia_delta]), cov = Ibc_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_Ia.rvs()
    cake_Ia = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = Ibc_Ia_var * np.eye(2))
    cake = normalize_hubble(cake_Ia.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), vb=vb)
    return cake

def fit_II(z, mu, vb=False):
    n_zs = len(z)
    n_mus = len(mu)

    #     cake = np.zeros((n_types, n_zs-1, n_mus-1))
    cake_Ia = sps.multivariate_normal(mean = np.array([z, II_Ia_delta]), cov = II_Ia_var * np.eye(2))
    [z_samp, mu_samp] = cake_Ia.rvs()
    cake_Ia = sps.multivariate_normal(mean = np.array([z_samp, mu_samp]), cov = II_Ia_var * np.eye(2))
    cake = normalize_hubble(cake_Ia.pdf(z_mu_grid.reshape(((n_zs-1)*(n_mus-1), 2))).reshape((n_zs-1, n_mus-1)), vb=vb)
    return cake



def fit_any(true_vals, vb=False):
    n_zs = len(z)
    n_mus = len(mu)

    #     print(unity_one)
    cake = np.zeros((n_types, n_zs-1, n_mus-1))#unity_one.copy()
    #     print(unity_one)
    if true_vals['t'] == 'Ia':
        cake[0] = fit_Ia(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[0]
    if true_vals['t'] == 'Ibc':
        cake[0] = fit_Ibc(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[1]
    if true_vals['t'] == 'II':
        cake[0] = fit_II(true_vals['z'], true_vals['mu'], vb=vb)
        ln_conf = ln_conf_matrix[2]
    if vb: print(np.exp(ln_conf))
    dist = sps.norm(loc = true_vals['z'], scale = z_sigma)
    z_means = dist.rvs(2)
    layer_Ibc = sps.norm(loc = z_means[0], scale = z_sigma).pdf(z_mids)
    #     print(layer_Ibc)
    layer_II = sps.norm(loc = z_means[1], scale = z_sigma).pdf(z_mids)
    #     print(layer_II)
    cake[1] = normalize_hubble(unity_hubble * layer_Ibc[:, np.newaxis], vb=vb)
    cake[2] = normalize_hubble(unity_hubble * layer_II[:, np.newaxis], vb=vb)
    #     cake = normalize_one(cake)
    if not np.all(cake>=0.):
        print(true_vals)
        assert False
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
