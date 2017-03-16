
from sklearn.svm import SVC
import numpy as np
import json
import yaml
import os
import pickle
import transform
from transforms import removevl

def load_and_filter_descriptors(filelist, filterdesc, preprocessing):
    """ Load a project filelist and perform transformations
        Returns:
            a dictionary of {filekey: filtereddata, ...}
    """

    ret = {}
    # TODO: If this doesn't exist?
    pre = preprocessing[filterdesc]
    print(pre)
    # TODO: If remove or include are first transforms, do them on load
    # Created a list of json-files and removed variable length params
    # The list is converted back into dictionary and passed for further transformations
    data_list = [(index, json.load(open(path))) for index, path in filelist.items()]
    data_list = removevl.RemoveVariableLength().transform(data_list)
    for k, data in data_list:
        tr = transform.transform(data, pre)
        ret[k] = tr

    # TODO: otherwise, do filtering after it's all loaded
    ret = transform.transform_all(ret, pre)

    return ret


def load_filelist(projectroot, flfile):
    return yaml.load(open(os.path.join(projectroot, flfile)))


def load_groundtruth(projectroot, gtfile):
    return yaml.load(open(os.path.join(projectroot, gtfile)))


def enumerate_combinations(project):
    params = []
    for s in project["classifiers"]["svm"]:
        for pre in s["preprocessing"]:
            for k in s["kernel"]:
                for g in s["gamma"]:
                    for c in s["C"]:
                        params.append({"preprocessing": pre, "kernel": k, "gamma": g, "C": c})
    return params


def train_model_iteration(projectroot, project, iteration):
    param_iters = enumerate_combinations(project)
    if iteration >= len(param_iters) or iteration < 0:
        raise Exception("iteration parameter is out of bounds of number of available iterations (%s)" % (len(param_iters), ))

    params = param_iters[iteration]
    print(params)

    flfile = project["filelist"]
    gt = project["groundtruth"]
    groundtruth = load_groundtruth(projectroot, gt)
    classes = groundtruth.values()

    print("Loading...")
    data = load_and_filter_descriptors(load_filelist(projectroot, flfile), params["preprocessing"], project["preprocessing"])
    print("done")

    # loop through data items, get all keys
    keys = set()
    for k, v in data.items():
        keys.update(set(v.keys()))

    # map gt class names to numbers
    class_map = dict([(v, i) for i, v in enumerate(classes, 1)]) # class labels to numbers
    keys = sorted(list(keys))
    numkeys = len(keys)


    numitems = len(data)
    feature_data = np.empty((numitems, numkeys))
    feature_classes = np.empty(numitems)

    for i, (k, d) in enumerate(data.items()):
        class_val = class_map[groundtruth[k]]
        feature_classes[i] = class_val
        # TODO: optimisation - make featuredata `numitems x numkeys`
        #       check how much memory this uses - will we replicate `data`?
        for j, featkey in enumerate(keys):
            # TODO: None or NaN?
            try:
                feature_data[i][j] = d.get(featkey)
            except ValueError:
                print(featkey)
                raise

    # TODO: We should do normalize/gaussianize here, because we have the data in a np array
    # and can use np methods to do it quickly

    print("verifying data")
    mbids = list(data.keys())
    for i in range(numitems):
        for j in range(numkeys):
            if np.isnan(feature_data[i][j]):
                print("item %s value %s is nan" % (mbids[i], keys[j]))
            if np.isinf(feature_data[i][j]):
                print("item %s value %s is inf" % (mbids[i], keys[j]))

    k = params["kernel"]
    g = params["gamma"]
    c = params["C"]
    clf = SVC(kernel=k, gamma=2**g, C=2**c)
    print("training model...")
    clf.fit(feature_data, feature_classes)
    print("done, dumping")

    # TODO: Also need to store the order of keys
    # TODO: and class->string mappings
    # Why not the entire dataset groundtruth?
    # Parameters
    # cross-validation accuracy
    # cross-validation confusion matrices
    pickle.dump(clf, open("test.pkl", "w"))
    print("done")

