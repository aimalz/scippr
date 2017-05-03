# SCIPPR

Supernova Cosmology Inference with Probabilistic Photometric Redshifts

## Motivation

Type Ia supernovae serve as standard candles that can be used to probe cosmological parameters.  Historically, small numbers of low-redshift supernovae were discovered in photometric data and immediately followed up with spectroscopic observations that could easily determine their redshifts and types with high confidence.  LSST will detect extraordinary numbers of transients at unprecedented redshifts, such that spectroscopic follow-up on any significant fraction of them will be impossible.  To harness the power of this newly accessible data, we must depend solely on photometry for the discovery, classification, and characterization of Type Ia supernovae.  Point estimates of supernova type and redshift that were reliable for spectroscopic data are inaccurate for this coarser data, but it may still be used!  Emerging techniques aim to use photometry to estimate probability distributions over supernova type and redshifts, but there remains a need for an end-to-end analysis code that performs inference of cosmological parameters from these high-dimensional data products without reducing them to traditional point estimates.  SCIPPR aims to satisfy this need!

## Usage

SCIPPR currently consists of two Jupyter notebooks, one to simulate some mock data in the form of interim posteriors and one to perform inference of the cosmological parameters used to generate that mock data.  These notebooks require the following packages: `astropy`, `daft`, `emcee`, `hickle`, `matplotlib`, `numpy`, `scipy`.

## People

This project is led by [Alex Malz](https://github.com/aimalz/scippr/issues/new?body=@aimalz)(NYU) and [Tina Peters](https://github.com/aimalz/scippr/issues/new?body=@tinapeters)(U.Toronto) in collaboration with [Anita Bahmanyar](https://github.com/Andromedanita)(U. Toronto), [Lluis Galbany](https://github.com/lgalbany)(U. Pitt), [Kara Ponder](https://github.com/kponder)(U. Pitt), and [Humna Awan](https://github.com/humnaawan)(Rutgers) and with support from [LSST-DESC](https://github.com/LSSTDESC).

## License, Contributing etc

The code in this repo is available for re-use under the MIT license, which means that you can do whatever you like with it, just don't blame us. If you end up using any of the code or ideas you find here in your academic research, please cite us as `Malz and Peters, et al., in preparation\footnote{\texttt{https://github.com/aimalz/scippr}}`. If you are interested in this project, please do drop us a line via the hyperlinked contact names above, or by [writing us an issue](https://github.com/aimalz/scippr/issues/new). To get started contributing to the `scippr` project, just fork the repo - pull requests are always welcome!
