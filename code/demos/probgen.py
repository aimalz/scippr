import pylab as pl
import numpy as np
import scipy.stats as st


def rejection_sampling(x, px, num=1000):
    from scipy.interpolate import interp1d

    samples = []
    pinterp = interp1d(x,px)
    max = np.max(px)
    
    
    number=0
    while number < num:
        z = np.random.uniform(0,1)
        u = np.random.uniform(0, 1/max)
                
        if u <= pinterp(z):
            samples.append(z)
            number+=1

    return np.array(samples)


step = 0.001
x = np.arange(0,1+step, step)
#px =  st.norm.pdf(x)

px = x

vals = rejection_sampling(x,px,num=5000)

pl.plot(x, px, 'r')
pl.savefig('test.png')
pl.clf()

pl.hist(vals)

pl.savefig('hi.png')
