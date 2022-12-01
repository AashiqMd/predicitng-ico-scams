from dict import Dictionary
from pathlib import Path
import torch
import math
from math import sqrt

import string

import pickle
from pathlib import Path
import hashlib
from os.path import exists

from datetime import datetime

DATA_DIR = Path.cwd() / "data" / "whitepapers"

CMK = DATA_DIR / "coinmarketcap"
SAPKOTA = DATA_DIR / "sapkota"
SRT = DATA_DIR / "srt32"
TXT = DATA_DIR / "text"

NOT_SCAM = "NotScam"
SCAM = "Scam"


pos_docs = list(CMK.glob("*.txt")) + list(SRT.glob("*.txt")) + list(TXT.glob("*.txt"))
neg_docs = list(SAPKOTA.glob("*.txt"))

CACHE = DATA_DIR / "cache"
cache_docs = list(CACHE.glob("*.txt"))
USE_CACHE = True

# just for dating an output file
date = datetime.now().strftime("%Y%m%d_%I%M%S")

# dictionary of my calculated classifications of each file
# 	ie. "coinmarketcap/bitcoin" : NotScam
results = {}
# distances_all_files = {}


def dot_prod(vector_1, vector_2):
	return float(sum(float(x) * float(y) for x, y in zip(vector_1, vector_2)))

def mag(vec):
	return float(sqrt(dot_prod(vec, vec)))

def cosine_similarity(vector_1, vector_2):
	# define the cosine similarity between the two vectors
	cosine_similarity = dot_prod(vector_1,vector_2) / ( mag(vector_1) * mag(vector_2) + 0.000000001 ) 
	return cosine_similarity


def score(file, distances_pos, distances_neg, this_file_distances):
	# cosine similarity -> highest number is more similar
	#	so we'll sort descending
	distances_pos.sort(reverse=True)
	# print(distances_pos)
	top_positive_distances = distances_pos[0:3]


	distances_neg.sort(reverse=True)
	# print(distances_neg)
	top_negative_distances = distances_neg[0:3]

	# select top 3 "nearest" neighbors in 'scam' set and in 'non-scam' set
	print(top_positive_distances)
	print("\tor")
	print(top_negative_distances)

	# previous model scoring method
	# # classify based on distances of neighbors
	# if sum(top_positive_distances) >= sum(top_negative_distances):
	# 	results[file] = NOT_SCAM
	# 	print("closer to non-scam coin whitepaper: %s" % file)

	# else:
	# 	results[file] = SCAM
	# 	print("SCAM coin (!): %s" % file)

	# simple k=3 method
	votes = []
	l = 0
	r = 0
	while l < len(top_positive_distances) and r < len(top_negative_distances):

		if len(votes) == 3:
			break

		if top_positive_distances[l] > top_negative_distances[r]:
			votes.append(NOT_SCAM)
			l += 1

		else:
			votes.append(SCAM)
			r += 1



	print(votes)
	if votes.count(SCAM) > votes.count(NOT_SCAM):
		results[file] = SCAM
	else:
		results[file] = NOT_SCAM



	# try instead of scoring it this way, just test for plagiarism, if anything has high similarity it is involved in plagiarism
	# if any (top_pos, top_neg) > 0.95:
	# 	then its involved in a scam, whichever was published second.
	for compare_to_file, cos_sim_score in this_file_distances.items():
		if cos_sim_score > 0.9:
			print("whitepaper is %s similar to %s" % (cos_sim_score, compare_to_file))


def get_distances_this_file(i, file, single_doc_vec, pos_subset_vecs, neg_subset_vecs, index_neg, index_pos):

	this_file_distances = {}
	# re-establish new distances for each single document 
	distances_pos = []
	distances_neg = []
	print("\nScoring the results for document %s... " % file)

	# if it has a super high score with these docs then thats weird.
	# compute the distances to all vectors in pos and neg
	for j in range(len(pos_subset_vecs)):

		# add to overall dict
		compare_to_file = index_pos[j]

		# get the cosine similarity
		cosine_sim_score = cosine_similarity(single_doc_vec, pos_subset_vecs[j])

		if not math.isnan(cosine_sim_score):
			distances_pos.append(cosine_sim_score)
			this_file_distances[compare_to_file] = cosine_sim_score

		if cosine_sim_score > 0.90:
			print("Similarity score > 0.90 with a doc which has a NON_SCAM label: %s" % j)
			print("document at pos_docs index i: %s similar to pos subset doc @ index j : %s" % (i,j))
			print(file)
			print(index_pos[j])
			# meeting the criteria here means that the coin plagiarises a NON_SCAM coin


	# if it has a super high score here means that it may have been plagiarised
	# do same for distances between this vector and the negative vectors
	for k in range(len(neg_subset_vecs)):

		compare_to_file = index_neg[k]

		# get the cosine similarity
		cosine_sim_score = cosine_similarity(single_doc_vec, neg_subset_vecs[k])

		if not math.isnan(cosine_sim_score):
			distances_neg.append(cosine_sim_score)
			this_file_distances[compare_to_file] = cosine_sim_score

		if cosine_sim_score > 0.90:
			print("Similarity score > 0.90 with a doc which has SCAM label: %s" % k)
			print("document at index i:%s similar to document index K : %s" % (i,k))
			print(file)
			print(index_neg[k])

	# SCORING
	score(file, distances_pos, distances_neg, this_file_distances)

	# save this dictionary of distances for the file
	return this_file_distances


def write_classes(results):
	out_filename = Path.cwd() / "data" / "out" / "out_knn.txt"
	with open(out_filename, "w") as outfile:
		for k,v in results.items():

			ostring = ("%s : %s" % (k,v))
			outfile.write(ostring)
			outfile.write("\n")

def write_object(file, distances_single_file):
	file_name = "out_all_distances_all_files_" + str(date) + ".txt"
	out_filename = Path.cwd() / "data" / "out" / file_name
	with open(out_filename, "a") as outfile:

		outfile.write("file1, file2, cosine_similarity\n")
		for q,r in distances_single_file.items():

			ostring = str(file) + "," + str(q) + "," + str(r)
			outfile.write(ostring)
			outfile.write("\n")


def file_in_cache(file, suffix):
	file = str(file)
	filename = hashlib.sha256(file.encode('utf-8')).hexdigest()
	filename = filename + suffix
	filepath = Path.cwd() / "data" / "cache" / filename
	return exists(filepath)


def persist_to_cache(file, this_dict, suffix):
	file = str(file)
	filename = hashlib.sha256(file.encode('utf-8')).hexdigest()
	filename = filename + suffix
	filepath = Path.cwd() / "data" / "cache" / filename

	with open(filepath, 'wb') as dictionary_file:
		pickle.dump(this_dict, dictionary_file)

def read_from_cache(file, suffix):
	file = str(file)
	filename = hashlib.sha256(file.encode('utf-8')).hexdigest()
	filename = filename + suffix
	filepath = Path.cwd() / "data" / "cache" / filename 

	print("reading file from cache %s" % filepath)
	with open(filepath, 'rb') as dictionary_file:
		return pickle.load(dictionary_file)


def prepare_with_cache_option():
	# 
	# PART 1 : chose one of the pos_docs as victim
	#
	# 	this goes through the pos docs picking one out at a time

	indexed_neg_set = {v: k for v, k in enumerate(neg_docs)}

	for i in range(len(pos_docs)):
		print("processing file %s/%s" % (i, len(pos_docs)))
		file = pos_docs[i]

		single_document = list()
		single_document.append(pos_docs[i])

		# subtract the single document from the rest of the docs
		pos_subset = [x for x in pos_docs if x not in single_document]
		indexed_pos_subset = {v: k for v, k in enumerate(pos_subset)}

		dict = Dictionary()
		neg_vecs = []
		pos_subset_vecs = []

		# LOAD THINGS FROM A CACHE

		if not USE_CACHE:
			print("building a dictionary of all positive subset and negative documents")

			for doc in pos_subset + neg_docs:
				dict.ingest_document(doc)
				persist_to_cache(file, dict, "_dict_cache.txt")

			# also make the larger tfidfs
			neg_vecs = [dict.get_tfidf(doc).tolist() for doc in neg_docs]
			pos_subset_vecs = [dict.get_tfidf(doc).tolist() for doc in pos_subset]

		else:
			# load dictionary
			if not file_in_cache(file, "_dict_cache.txt"):
				# print("file not in the cache, need to add it...")
				for doc in pos_subset + neg_docs:
					dict.ingest_document(doc)
					persist_to_cache(file, dict, "_dict_cache.txt")
				print("done creating dict and to cache")
			else:
				dict = read_from_cache(file, "_dict_cache.txt")

			# load pos_subset_vecs
			if not file_in_cache(file, "_posvec_cache.txt"):
				pos_subset_vecs = [dict.get_tfidf(doc).tolist() for doc in pos_subset]
				persist_to_cache(file, pos_subset_vecs, "_posvec_cache.txt")
				print("done adding pos vec for file to cache")
			else:
				pos_subset_vecs = read_from_cache(file, "_posvec_cache.txt")

			# load neg_vecs
			if not file_in_cache(file, "_negvec_cache.txt"):
				neg_vecs = [dict.get_tfidf(doc).tolist() for doc in neg_docs]
				persist_to_cache(file, neg_vecs, "_negvec_cache.txt")
				print("done adding pos vec for file to cache")
			else:
				neg_vecs = read_from_cache(file, "_negvec_cache.txt")


		# tf idf of the single document
		print("creating tf idf for single doc item ...")
		single_document_vec = [dict.get_tfidf(doc).tolist() for doc in single_document]
		single_doc_vec = single_document_vec[0]

		distances_from_this_file = get_distances_this_file(i, pos_docs[i], single_doc_vec, pos_subset_vecs, neg_vecs, indexed_neg_set, indexed_pos_subset )

		# save results to file as you continue
		write_object(file, distances_from_this_file)
		write_classes(results)


	# 
	# PART 2 : chose one of the NEG_docs as victim
	#
	# this goes through the negative docs picking one out at a time

	indexed_pos_set = {v: k for v, k in enumerate(pos_docs)}

	print("\n\nMoving on to picking out one of the negative docs...")
	for i in range(len(neg_docs)):

		print("processing file %s/%s" % (i, len(neg_docs)))
		file = neg_docs[i]
		
		single_document = list()
		single_document.append(neg_docs[i])

		# subtract the single document from the rest of the docs
		neg_subset = [x for x in neg_docs if x not in single_document]
		indexed_neg_subset = {v: k for v, k in enumerate(neg_subset)}

		dict = Dictionary()
		neg_subset_vecs = []
		pos_vecs = []

		# LOAD THINGS FROM A CACHE

		if not USE_CACHE:
			print("building a dictionary of all positive subset and negative documents")
			
			for doc in neg_subset + pos_docs:
				dict.ingest_document(doc)
				persist_to_cache(file, dict, "_dict_cache.txt")

			# also make the larger tfidfs
			neg_subset_vecs = [dict.get_tfidf(doc).tolist() for doc in neg_subset]
			pos_vecs = [dict.get_tfidf(doc).tolist() for doc in pos_docs]

		else:
			# load dictionary
			if not file_in_cache(file, "_dict_cache.txt"):
				# print("file not in the cache, need to add it...")
				for doc in neg_subset + pos_docs:
					dict.ingest_document(doc)
					persist_to_cache(file, dict, "_dict_cache.txt")
				print("done creating dict and to cache")
			else:
				dict = read_from_cache(file, "_dict_cache.txt")

			# load pos_vecs
			if not file_in_cache(file, "_posvec_cache.txt"):
				pos_vecs = [dict.get_tfidf(doc).tolist() for doc in pos_docs]
				persist_to_cache(file, pos_vecs, "_posvec_cache.txt")
				print("done adding pos vec for file to cache")
			else:
				pos_vecs = read_from_cache(file, "_posvec_cache.txt")


			# load neg_subset_vecs
			if not file_in_cache(file, "_negvec_cache.txt"):
				neg_subset_vecs = [dict.get_tfidf(doc).tolist() for doc in neg_subset]
				persist_to_cache(file, neg_subset_vecs, "_negvec_cache.txt")
				print("done adding pos vec for file to cache")
			else:
				neg_subset_vecs = read_from_cache(file, "_negvec_cache.txt")


		# tf idf of the single document
		print("creating tf idf for single doc item ...")
		single_document_vec = [dict.get_tfidf(doc).tolist() for doc in single_document]
		single_doc_vec = single_document_vec[0]

		# save for later
		distances_from_this_file = get_distances_this_file(i, neg_docs[i], single_doc_vec, pos_vecs, neg_subset_vecs, indexed_neg_subset, indexed_pos_set )

		# save results to file as you continue
		write_object(file, distances_from_this_file)
		write_classes(results)


if __name__ == "__main__":
	prepare_with_cache_option()
	write_classes(results)


























