"""
tsukuyomi microservice
The python implementation of helpful and somehow generic functions

Created: April 2022
@author: Willi Kristen

@license: Willi Kristen:
Copyright (c) 2022 Willi Kristen, Germany
https://de.linkedin.com/in/willi-kristen-406887218

All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution.
This software is the confidential and proprietary information of Willi Kristen.
You shall not disclose such confidential information and shall use it only in accordance with the
terms of the license agreement you entered into with Willi Kristen's software solutions.
"""

import gc
import json
import numpy as np
import pandas as pd
import pickle
import ssl

from datetime import datetime as dt
from importlib import import_module
from itertools import chain, combinations
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm_batch.batch_process import batch_process

from tsukuyomi.utils.util_tools import CONFIG_PATH_SCRIPT, load_config
from tsukuyomi.business_layer.data.data_generation import DataGenerator
from tsukuyomi.business_layer.data.filter_generation import FilterGenerator
from tsukuyomi.business_layer.preprocessing.preprocessing import TsukuyomiTransformer
from tsukuyomi.presentation_layer.data_model import TrainingItem
from tsukuyomi.business_layer.features.feature_generation import TsukuyomiFeatures

ssl._create_default_https_context = ssl._create_unverified_context

PATTERN_TIMESTAMP = "%d.%m.%Y, %H:%M:%S"

config = load_config(CONFIG_PATH_SCRIPT)

cores_involved = config['PERFORMANCE']['CORES_INVOLVED']
cv_splits = config['CROSS_VALIDATION']['SPLITS']
cv_test_size = config['CROSS_VALIDATION']['TEST_SIZE']

# variables to enable certain parts of the script
write_data = config['PART_ENABLING']['WRITE_DATA']
create_stopwords = config['PART_ENABLING']['CREATE_FILTERS']
check_formals = config['PART_ENABLING']['CHECK_FORMALS']
write_processed_data = config['PART_ENABLING']['WRITE_PROCESSED_DATA']
feature_combinations = config['PART_ENABLING']['FEATURE_COMBINATIONS']
write_tuning_res = config['PART_ENABLING']['WRITE_TUNING_RESULTS']

# variables for time measurements
start_script = None
start_stage = None

start_script = dt.now()
start_stage = dt.now()
print(f"\n[START SCRIPT] @{start_script.strftime(PATTERN_TIMESTAMP)}")
print(f"\n[START STAGE]: 'Create Data Depot'")
print("\tBeginning to fetch the XML documents via URL-path...\n")

dg = DataGenerator()
data = dg.create_model_data()

if write_data:
    print("\tBeginning to write the data into data depot (archive/data_depot/*)..\n")
    for idx, elem in enumerate(data):
        with open(f"archive/data_depot/bt_extracted_speech_{idx+1}.json", "w") as j:
            json.dump(elem, j)

print(f"[STAGE END] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
print(f"\tStage found {len(data)} speeches. Data Depot created in {dt.now() - start_stage}.\n\n")

if create_stopwords:
    start_stage = dt.now()
    print(f"\n[START STAGE]: 'Create Filter' @{start_stage.strftime(PATTERN_TIMESTAMP)}")
    print("\tBeginning to fetch the XML document for extracting relevant representatives...\n")

    fg = FilterGenerator()
    reps, adds, norms = fg.create_stopwords()
    print(f"\n\n\tData will be filtered for:\n\t==========================\n" \
          f"\t● {len(reps):>{6}} representatives\n"\
          f"\t● {len(adds):>{6}} additional parliamentarian terms\n"\
          f"\t● {len(norms):>{6}} german stop words\n")

    if check_formals:
        fg.check_fomals(data)

    print(f"[STAGE END] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
    print(f"\tStage created all files containing the necessary filters for the preprocessing in: {dt.now() - start_stage}\n\n")
else:
    print(f"[STAGE] 'Preprocessing Filters' @{dt.now().strftime(PATTERN_TIMESTAMP)}")
    print("\tAll the filters needed for the preprocessing are already created and have been loaded...\n")

start_stage = dt.now()
print(f"\n[START STAGE] 'Preprocessing' @{start_stage.strftime(PATTERN_TIMESTAMP)}")
print("\tBeginning to map the speeches to data model...\n")

data = [TrainingItem.construct(**elem) for elem in data]
data_count = len(data)

print("\tBeginning to process and filter each speech in data set...\n")

tt = TsukuyomiTransformer()
data = batch_process(items=[[item] for item in data], function=tt.transform, n_workers=cores_involved)
data = [item for sublist in data for item in sublist]
# data = Parallel(n_jobs=4, backend="loky")(delayed(tt.transform)([speech]) for speech in tqdm(data))

print(f"[END STAGE] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
print(f"\tStage preprocessed text data of {len(data)} speeches in: {dt.now() - start_stage}.\n")

start_stage = dt.now()
print(f"\n[START STAGE] 'Feature Extraction' @{start_stage.strftime(PATTERN_TIMESTAMP)}")
print("\n\ty:\n\t\t[INFO] Converting the names of the political parties in numeric labels.\n")

y = []
for item in data:
    match item.party:
            case "CDU/CSU":
                y.append(0)
            case "SPD":
                y.append(1)
            case "AfD":
                y.append(2)
            case "FDP":
                y.append(3)
            case "BÜNDNIS 90/DIE GRÜNEN":
                y.append(4)
            case "DIE LINKE":
                y.append(5)

y = np.array(y)

print("\n\tInitialization of features started...\n")

tf = TsukuyomiFeatures()
X = tf.fit_transform(data)

feature_count = X.shape[1]

if write_processed_data:
    print("\tBeginning to write the processed data into data depot (archive/data_depot/*)..\n")
    for idx, elem in enumerate(data):
        with open(f"archive/data_depot/bt_processed_speech_{idx+1}.json", "w") as j:
            json.dump(elem, j)

del data
gc.collect()
print("\n\t[INFO] Preprocessed data is not needed anymore... cleared from memory!\n")

scaler = StandardScaler()
X = scaler.fit_transform(X)

with open("./tsukuyomi/business_layer/models/scaler.pkl", 'wb') as p:
    pickle.dump(scaler, p)

print(f"[END STAGE] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
print(f"\tStage generated feature data, set up the necessary models and scaled the feature output: {dt.now() - start_stage}.\n" )

start_stage = dt.now()
print(f"\n[START STAGE] 'Training & Tuning' @{start_stage.strftime(PATTERN_TIMESTAMP)}")
print("\n\tPreparing different combinations of feature data for the tuning...\n")

if feature_combinations:
    feature_dict = {}
    previous = 0
    for name, transformer in tf.feature_generators_:
        current = (previous + transformer.no_features)
        feature_dict[name] = X[:, previous:current]
        previous = current

    feature_combos = chain(*map(lambda x: combinations(list(feature_dict.keys()), x), range(0, len(list(feature_dict.keys()))+1)))
    feature_combos = list(feature_combos)[-1:] #[29:] # From combinations of 4 and higher

    features = {}
    for combo in feature_combos:
        name = ""
        feat_concat = None
        for feature in combo:
            if name == "":
                name = feature
            else:
                name += ", " + feature
            if feat_concat is None:
                feat_concat = feature_dict[feature]
            else:
                feat_concat = np.concatenate((feat_concat, feature_dict[feature]), axis=1)
        features[name] = feat_concat

sss = StratifiedShuffleSplit(n_splits=cv_splits, test_size=cv_test_size, random_state=99)

active_models: list = config['MODELS']['ACTIVE']

models = {}
params = {}
for model in active_models:
    module = config['MODELS'][model]['MODULE']
    name = config['MODELS'][model]['NAME']
    first_iter_raw = config['MODELS'][model]['PARAMS']['FIRST_ITER']

    clf = getattr(import_module(module), name)
    clf = clf()
    models[name] = clf

    first_iter = {}
    for k, v_raw in first_iter_raw.items():
        v = []
        type_cast = v_raw.pop(0)
        match type_cast:
            case "str":
                for p in v_raw:
                    p = str(p)
                    v.append(p)
                first_iter[k] = v
            case "int":
                for p in v_raw:
                    p = int(p)
                    v.append(p)
                first_iter[k] = v
            case "float":
                for p in v_raw:
                    p = float(p)
                    v.append(p)
                first_iter[k] = v
    params[name] = first_iter

combo_count = len(feature_combos) if feature_combinations else 1
model_count = len(models)
params_count = sum([len(v) for _, v in params.items()])

estimated_time_per_fit = 37.33
total_fits = (combo_count * cv_splits * model_count * params_count)
hours = (total_fits * 37.33 / 60 / 60)
days = (hours / 24)

print("\t[INFO] - TUNING OVERVIEW:\n")
print("\t\tOverview of estimated time for tuning and validation of the model candidates and params:")
print("\t\t----------------------------------------------------------------------------------------\n\n")

print(f"\t\tModels to test:.................................................................{model_count}\n")
print(f"\t\tSplits in cross validation:.....................................................{cv_splits}\n")
print(f"\t\tFeature combos to test:.........................................................{combo_count}\n")
print(f"\t\tParameters to tune:.............................................................{params_count}\n")

print("\n\t\t========================================================================================\n")
print(f"\t\t\tEstimated time:\t\t\t{round(hours, 2)}h - {round(days, 2)} days\n")

idx = 0
tuning = {}
if feature_combinations:
    for k in models:
        model = models[k]
        model_params = params[k]
        for combo, feat in features.items():
            idx += 1
            tuning[f"tuning_{idx}"] = {"name": k, 
                                       "model": model,
                                       "params": model_params,
                                       "combo": combo,
                                       "data": feat}
else:
    for k in models:
        idx += 1
        tuning[f"tuning_{idx}"] = {"name": k, 
                                   "model": models[k],
                                   "params": params[k],
                                   "combo": "All Features",
                                   "data": X}

tuning_start = dt.now()
print(f"\n\t[TUNING STARTED] First iteration @{tuning_start.strftime(PATTERN_TIMESTAMP)}\n")
for t, tuning_set in tuning.items():
    model = tuning_set['model']
    combo = tuning_set['combo']
    params = tuning_set['params']
    data = tuning_set['data']

    print(f"\n\t\t[Start] Tuning for:\n\t\t..model:.....{model}\n\t\t..combo:.....{combo}\n")
    grid = GridSearchCV(model, param_grid=params, n_jobs=cores_involved, cv=sss)
    grid.fit(data, y)
    tuning[t] = (tuning_set, grid)

print(f"\n\tTuning duration for {total_fits} fits:\n")
print(f"\t{dt.now() - tuning_start}\n")

print("\tChoosing best model and combination of features:\n")
print("\t(If you would like to have a more detailed look into the tuning results, check:\n\t\t./setup_files/tuning_result.pkl)\n")
dfs_concat = []
for t, v in tuning.items():
    tuning[t] = {"tuning": tuning[t][0],
                 "grid": tuning[t][1],
                 "display": pd.DataFrame(v[1].cv_results_)[['params', 'mean_test_score', 'std_test_score']]}
    df_tmp = tuning[t]['display']
    df_tmp['tuning_code'] = t
    df_tmp['model'] = tuning[t]['tuning']['name']
    df_tmp['combo'] = tuning[t]['tuning']['combo']
    dfs_concat.append(df_tmp)

result_df = pd.concat(dfs_concat)
result_df.sort_values(by='mean_test_score', ascending=False, inplace=True)
result = result_df.iloc[0].to_dict()

selected = tuning[result['tuning_code']]

selected_model = selected['grid'].best_estimator_
data = selected['tuning']['data']

second_iter = {}
for k, v in config['MODELS'].items():
    if k == "ACTIVE":
        continue
    if result['model'] == v['NAME']:
        for k, v_raw in v['PARAMS']['SECOND_ITER'].items():
            v = []
            type_cast = v_raw.pop(0)
            match type_cast:
                case "str":
                    for p in v_raw:
                        p = str(p)
                        v.append(p)
                    second_iter[k] = v
                case "int":
                    for p in v_raw:
                        p = int(p)
                        v.append(p)
                    second_iter[k] = v
                case "float":
                    for p in v_raw:
                        p = float(p)
                        v.append(p)
                    second_iter[k] = v

if write_tuning_res:
    print("\tBeginning to write the result set of the first tuning iteration (scripts/setup_files/tuning_result.pkl)..\n")
    for k, v in tuning.items():
        del tuning[k]['tuning']['data']
    with open("./scripts/setup_files/tuning_result.pkl", "wb") as p:
        pickle.dump(tuning, p)

del tuning, result_df
gc.collect()
print("\t[INFO] Results of first tuning iteration are not needed anymore... cleared from memory!\n")

tuning_start = dt.now()
print(f"\n\t[TUNING STARTED] Second iteration @{tuning_start.strftime(PATTERN_TIMESTAMP)}\n")

print(f"\n\t\t[Start] Tuning for:\n\t\t..model:.....{selected_model}\n\t\t..combo:.....{selected['tuning']['combo']}\n")
grid = GridSearchCV(selected_model, param_grid=second_iter, n_jobs=cores_involved, cv=sss)
grid.fit(data, y)

print(f"\n\tTuning duration for second tuning iteration:\n")
print(f"\t{dt.now() - tuning_start}\n")

display = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'std_test_score']]

final_model = grid.best_estimator_
final_model.fit(data, y)
final_features = selected['tuning']['combo']
final_score = round((display['mean_test_score'].values[0] * 100), 2)
model_type = str(type(final_model)).split(".")[-1][:-2]
print(f"\tFinal Score, reached by {model_type}:\t{final_score}%\n")
print(f"\tFinal features combination:\t{final_features}.\n")

print("\tSaving final model and features combination..\n")
with open("./tsukuyomi/business_layer/models/service_clf.pkl", "wb") as p:
    pickle.dump(final_model, p)
with open("./tsukuyomi/business_layer/models/active_features.pkl", "wb") as p:
    pickle.dump(final_features, p)

print(f"[END STAGE] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
print(f"\tStage selected a model, trained it and found the optimal hyperparameters in: {dt.now() - start_stage}.\n" )

print(f"\n[END SCRIPT] @{dt.now().strftime(PATTERN_TIMESTAMP)}")
print(f"\tScript processed {data.shape[0]} speeches and calculated the optimum of {data.shape[1]} features for production in: {dt.now() - start_script}.\n" )
