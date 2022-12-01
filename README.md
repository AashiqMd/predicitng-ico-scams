# Predicting ICO scams

Code for predicting ICO scams based on cryptocurrency metadata.

To get started:
- get a download code from Alex then run download.py
- run preprocess.py to process the data

## Howell Dataset
Sabrina T Howell, Marina Niessner, David Yermack, Initial Coin Offerings: Financing Growth with Cryptocurrency Token Sales, The Review of Financial Studies, Volume 33, Issue 9, September 2020, Pages 3925–3974, https://doi.org/10.1093/rfs/hhz131

## Sapkota Dataset
Sapkota, Niranjan and Grobys, Klaus and Dufitinema, Josephine, How Much Are We Willing To Lose in Cyberspace? On the Tail Risk of Scam in the Market for Initial Coin Offerings (November 18, 2020). http://dx.doi.org/10.2139/ssrn.3732747

## RapidMiner
Aashiq hand-cleaned a version of the dataset to use with RapidMiner. Get the right ID from Alex, then you can load the dataset in and begin experimenting there.

## Pytorch
To run the pytorch code, you'll need to use the `hydra_main.py` file. This sets up the configuration (using hydra rather than argparse).

Install pytorch with the right command from https://pytorch.org/.