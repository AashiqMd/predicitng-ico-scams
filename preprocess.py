#!/usr/bin/python3

from pathlib import Path
import data_properties

import logging
import pandas as pd

"""
	USAGE:
		(1) Separately/manually download the HOWELL dataset using alex's download program
		(2) Separately/manually download the SAPOKTA dataset using alex's download program
		(3) pip3 install -r requirements.txt
		(4) python3 preprocess.py

"""

NOT_SCAM = "NotScam"

logging.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=logging.INFO,
	datefmt='%Y-%m-%d %H:%M:%S')
logger = logging

DATA_DIR = Path.cwd() / 'data'

howell_path_stata = DATA_DIR / 'master_data_07142019_a.dta'
howell_path = DATA_DIR / 'master_data_07142019_a.csv'

sapkota_path_excel = DATA_DIR / 'sapkota' / 'Scam100.xlsx'

reduced_merged_data = DATA_DIR / "final_merged_data.csv"


def convert_without_losing_NAs():
	""" Simply converts the file (on disk) to CSV format
	"""
	data = pd.io.stata.read_stata(howell_path_stata, convert_missing=False)
	data.to_csv(howell_path)


def preprocess_howell(log=True):
	""" Selects 137 columns out of the original 310 available cols
		Mostly the fields with "last date of" and pricing info was dropped

		Returns:
			A dataframe with reduced column list
	"""
	if log:
		logger.info("printing raw fields list")
		logger.info(data_properties.howell_raw_fields)

		logger.info("Using only the following fields from the Howell dataset:")
		logger.info(data_properties.howell_fields)

	howell_df = pd.read_csv(howell_path, header='infer', usecols=data_properties.howell_fields)
	howell_df = howell_df.replace(r'\n',' ', regex=True) 

	if log:
		logger.info("printing howell sample ...")
		logger.info(howell_df)

	return howell_df


def preprocess_sapkota(log=True):
	""" This one is more simple
		There is just one XLS which we convert to dataframe
			Takes 20 columns out of the original 41 fields

		Returns:
			A dataframe with reduced column list
	"""
	sapkota_df = pd.read_excel(sapkota_path_excel, usecols=data_properties.sapkota_fields)
	if log:
		logger.info("printing sapkota SCAM100 sample ...")
		logger.info(sapkota_df)

	return sapkota_df



def merge_left(howell, sapkota, how="left", log=True):
	assert len(howell) == 1520, f"unexpected size for howell dataset: {len(howell)}"
	assert len(sapkota) == 100, f"unexpected size for sapkota dataset: {len(sapkota)}"

	# add source name as a column
	howell['data source'] = "HOWELL"
	sapkota['data source'] = "SAPKOTA"

	howell['token_name'] = howell['token_name'].str.lower()
	sapkota['Name'] = sapkota['Name'].str.lower()
	matches_df = pd.merge(howell, sapkota, left_on="token_name", right_on="Name", how=how)
	
	matches_df["Type"].fillna(NOT_SCAM, inplace=True)

	if log:
		logger.info("These coins overlap! We can combine them to a single dataset")
		print(matches_df)
	# okay cool so there is some overlap in the coins
	# 83 coins from sapkota scam100 file are in the howell dataset

	matches_df.to_csv(DATA_DIR / 'merged_data_left_join.csv')



def merge(howell, sapkota, how="inner", log=True):
	assert len(howell) == 1520, f"unexpected size for howell dataset: {len(howell)}"
	assert len(sapkota) == 100, f"unexpected size for sapkota dataset: {len(sapkota)}"

	# add source name as a column
	howell['data source'] = "HOWELL"
	sapkota['data source'] = "SAPKOTA"

	howell['token_name'] = howell['token_name'].str.lower()
	sapkota['Name'] = sapkota['Name'].str.lower()
	matches_df = pd.merge(howell, sapkota, left_on="token_name", right_on="Name", how=how)
	
	matches_df["Type"].fillna(NOT_SCAM, inplace=True)

	if log:
		logger.info("These coins overlap! We can combine them to a single dataset")
		print(matches_df)
	# okay cool so there is some overlap in the coins
	# 83 coins from sapkota scam100 file are in the howell dataset

	matches_df.to_csv(DATA_DIR / 'merged_data.csv')
	return matches_df


def load_final_merged():
	return pd.read_csv(reduced_merged_data)


if __name__ == '__main__':
	logger.info("starting with howell....")
	convert_without_losing_NAs()
	howell_df = preprocess_howell()

	logger.info("finished with initial howell. will continue with sapkota..")

	sapkota_scam_100_df = preprocess_sapkota()
	logger.info("done processing sapkota scam100 excel file.")
	
	# next, check if there is any overlap in coins between the two sets
	logger.info("checking if there is any overlap in coins between the files")

	# write left join file
	merge_left(howell_df, sapkota_scam_100_df)

	# write inner join file
	merge(howell_df, sapkota_scam_100_df)





