# -*- coding:utf-8 -*-

from sklearn.cluster import KMeans
from retrieval import load_feat_db
from sklearn.externals import joblib
from config import DATASET_BASE, N_CLUSTERS
import os
import argparse


# if __name__ == '__main__':
#     feats, labels = load_feat_db()
#     model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
#     model_path = os.path.join(DATASET_BASE, r'models', r'kmeans.m')
#     joblib.dump(model, model_path)

# Modified version for scrapped data
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--scrapped", help="run kmeans on scrapped dataset rather than on deepfashion", 
		action="store_true")
	args = parser.parse_args()
	if args.scrapped:
		print("Performing kmeans clustering on scrapped dataset.")
		# feats, labels = load_feat_db(custom=True)
		feats, color_feats, labels = load_feat_db(custom=True)
		model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
		model_path = os.path.join(DATASET_BASE, r'models', r'kmeans_scrapped.m')
		joblib.dump(model, model_path)

	else:
		print("Performing kmeans clustering on deepfashion dataset.")
		# feats, labels = load_feat_db()
		feats, color_feats, labels = load_feat_db()
		model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
		model_path = os.path.join(DATASET_BASE, r'models', r'kmeans.m')
		joblib.dump(model, model_path)


	# feats, labels = load_feat_db()
	# model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
	# model_path = os.path.join(DATASET_BASE, r'models', r'kmeans.m')
	# joblib.dump(model, model_path)
