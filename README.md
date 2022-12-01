# Predicting ICO scams

Code for predicting ICO scams based on cryptocurrency metadata.

## Installation

First, run `pip install -r requirements.txt`.

Then, install pytorch with the right command for your machine from https://pytorch.org/.

You'll also need to open a python terminal and run...
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
```

## Running the KNN code

To run the K Nearest Neighbors model, you'll run `python3 whitepapers_single_x.py`. This is the code for the nearest neighbors model. It is automatically set to use cached files for the TF-IDF components in order to improve the runtime of this model. It is currently set to use the simple K=3 model which we observed better performance with. By using the cached files the entire results set can be computed in < 30 mins. 

Results for this model can be found in `data/out`. Two files will be produced by this model. 
- First, will be a comprehensive file containing the distance matrix for every file in the dataset (`out_all_distances_all_files.txt`)

- Second, will be an abbreviated file with the top-level classification results of each file, based on the KNN implementation.
For every file removed from the set, and KNN classification performed on the rest of the set (for every file), you will see:
`{filename} : {predicted label}` (`out_knn.txt`)

## Running the PyTorch code

To launch the code, you'll need to use the `hydra_main.py` file. This sets up the configuration (using hydra rather than argparse).

To reproduce the results of the metadata features task, you can run the following:
```
python hydra_main.py task=full model=lin num_feats=18 num_epochs=150 valid_every_n=5
python hydra_main.py task=full model=linb num_feats=18 num_epochs=150 hdim=128 num_layers=1  valid_every_n=5
python hydra_main.py task=full model=mlp num_feats=18 num_epochs=150 hdim=64 num_layers=1  valid_every_n=5
```

To reproduce the results of the whitepaper features task:
```
python hydra_main.py task=paper_sents num_epochs=25 tokenizer=re model=bow edim=16 hdim=16 dropout=0.5
python hydra_main.py task=paper_sents num_epochs=25 tokenizer=re model=rnn edim=32 hdim=32 dropout=0.5
```

You can see some of the other experimental results for the whitepaper task in the experiments.md file.

All of these commands run better on CPUs except for the RNN model which runs better on a GPU - you can let it run for just a few epochs on a CPU instead waiting for all 25 epochs to see un-converged results, though.

We select final models using max f1 as the criteria.

Highest / lowest-ranked sentences by scam-score are manually inspected using the final lines of train_nn.py. If you would like to do this yourself, you'll need to uncomment that code then wait for training to complete (ie. with the first whitepaper model, the bow model). Note that we throw out the sentences that look like hash values: they're actually a badly translated pdf file, which could be cleaned from the dataset if experimentation continued.


# Additional notes

We use limited versions of the following two datasets, plus data we collected ourselves.

**Howell Dataset**
Sabrina T Howell, Marina Niessner, David Yermack, Initial Coin Offerings: Financing Growth with Cryptocurrency Token Sales, The Review of Financial Studies, Volume 33, Issue 9, September 2020, Pages 3925–3974, https://doi.org/10.1093/rfs/hhz131

**Sapkota Dataset**
Sapkota, Niranjan and Grobys, Klaus and Dufitinema, Josephine, How Much Are We Willing To Lose in Cyberspace? On the Tail Risk of Scam in the Market for Initial Coin Offerings (November 18, 2020). http://dx.doi.org/10.2139/ssrn.3732747

We commit to the repo itself only data that we actually used from these datasets as we did not have any permissions to distribute it, but you can find code (e.g. in the download and preprocess and data_properties files) which contain efforts to process different parts of the code which ultimately are not part of our final numbers and so do not need to be reproduced.

## RapidMiner
Aashiq hand-cleaned a version of the Sapkota dataset to use with RapidMiner, which is not committed to the repo because it doesn't have any downstream dependencies.

## Reddit
We scraped a lot of data from reddit and originally intended to match coins to reddit posts. We found the whitepaper approach to be more promising, but you can find reddit code in the pushshift_download and parse_reddit files.
