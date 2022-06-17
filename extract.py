# -*- coding: utf-8 -*-
"""Extract data from solved Pulp model."""

___author__ = "2022 Ulmer Albert"

import datetime as dt
import pandas as pd


def extractVariablesDataframe(model, label='none', runningdate=str(dt.datetime.now())):

    result0 = pd.DataFrame(data=None)
    result1 = pd.DataFrame(data=None)
    result2 = pd.DataFrame(data=None)
    result3 = pd.DataFrame(data=None)

    for v in model.variables():
        namesplit = v.name.split('_')
        basename = namesplit[0]
        #print(basename, len(namesplit))

        if len(namesplit) == 1:  # single value
            result0.loc[0, basename] = v.varValue
        elif len(namesplit) == 2:  # simple index
            if namesplit[1].isnumeric():
                indexval = int(namesplit[1])
                result1.loc[indexval, basename] = v.varValue
        elif len(namesplit) == 3:  # tuple index 2 values
            indexval = ''.join(namesplit[1:])
            if indexval[0] == '(' and indexval[-1] == ')':
                result2.loc[indexval, basename] = v.varValue
        elif len(namesplit) == 4:  # tuple index 3 values
            indexval = ''.join(namesplit[1:])
            if indexval[0] == '(' and indexval[-1] == ')':
                result3.loc[indexval, basename] = v.varValue
        else:
            print('error!!')

    if len(result0) > 0:
        result0['label'] = label
        result0['runningdate'] = pd.to_datetime(runningdate)

    if len(result1) > 1:
        result1['label'] = label
        result1['runningdate'] = pd.to_datetime(runningdate)

    # convert single tuple-string index into proper multiindex
    if len(result2) > 1:
        result2['tmpindex'] = result2.index
        result2['tmpindex2'] = result2.tmpindex.apply(lambda x: eval(x))
        result2.index = pd.MultiIndex.from_tuples(
            result2.tmpindex2)  # , names=['vehicle', 'period'])
        result2.drop(['tmpindex', 'tmpindex2'], axis=1, inplace=True)
        result2['label'] = label
        result2['runningdate'] = pd.to_datetime(runningdate)

    if len(result3) > 1:
        result3['tmpindex'] = result3.index
        result3['tmpindex2'] = result3.tmpindex.apply(lambda x: eval(x))
        result3.index = pd.MultiIndex.from_tuples(
            result3.tmpindex2)  # , names=['vehicle', 'period', 'scenario'])
        result3.drop(['tmpindex', 'tmpindex2'], axis=1, inplace=True)
        result3['label'] = label
        result3['runningdate'] = pd.to_datetime(runningdate)

    return result0.sort_index(), result1.sort_index(), result2.sort_index(), result3.sort_index()


# def extractNamedVariables(model):
#     # return {v.getAttr("VarName"): v for v in model.getVars()}
#     # return model.variables()
#     varsdict = {}
#     for v in model.variables():
#         varsdict[v.name] = v.varValue
#     return varsdict


# def extractNamedVariablesWithBaseName(model, name):
#     return {
#         n: v
#         for (n, v) in extractNamedVariables(model).items()
#         # if n.startswith(name + "[") or n == name
#         if n.startswith(name + "_") or n == name
#     }


# def extractVariablesWithBaseName(model, name):
#     return [
#         v
#         for (n, v) in extractNamedVariablesWithBaseName(model, name).items()
#     ]


# def extractOptimalValueDict(model, name):
#     return {
#         n: v.getAttr("X")
#         for (n, v) in extractNamedVariablesWithBaseName(model, name).items()
#     }


# # def extractOptimalValueArray(model, name):
# #     # We assume that variables returned from `extractVariablesWithBaseName`
# #     # are correctly sorted by the time index!
# #     data = [
# #         v.getAttr("X")
# #         for v in extractVariablesWithBaseName(model, name)
# #     ]
# #     return np.array(data)


# def extractOptimalValueArray(model, name):
#     # We assume that variables returned from `extractVariablesWithBaseName`
#     # are correctly sorted by the time index!

#     # fetch all variables into dictionary
#     varsdict = {}
#     for v in model.variables():
#         varsdict[v.name] = v.varValue

#     return varsdict

#     # data = [
#     #     v.getAttr("X")
#     #     for v in extractVariablesWithBaseName(model, name)
#     # ]
#     # return np.array(data)


# def extractReducedCostArray(model, name):
#     # We assume that variables returned from `extractVariablesWithBaseName`
#     # are correctly sorted by the time index!
#     data = [
#         v.getAttr("RC")
#         for v in extractVariablesWithBaseName(model, name)
#     ]
#     return np.array(data)


# def extractNamedConstraints(model):
#     return {c.getAttr("ConstrName"): c for c in model.getConstrs()}


# def extractDualValuesDict(model):
#     return {
#         n: c.getAttr("Pi")
#         for (n, c) in extractNamedConstraints(model).items()
#     }


# def extractDualValueDict(model, name):
#     return {
#         n: c.getAttr("Pi")
#         for (n, c) in extractNamedConstraints(model).items()
#         if n.startswith(name + "[") or n == name
#     }


# def extractDualValueArray(model, name):
#     # We assume that variables returned from `extractNamedConstraints`
#     # are correctly sorted by the time index!
#     data = [
#         c.getAttr("Pi")
#         for (n, c) in extractNamedConstraints(model).items()
#         if n.startswith(name + "[") or n == name
#     ]
#     return np.array(data)


# def extractSlackValueDict(model, name):
#     return {
#         n: c.getAttr("Slack")
#         for (n, c) in extractNamedConstraints(model).items()
#         if n.startswith(name + "[") or n == name
#     }


# def extractSlackValueArray(model, name):
#     # We assume that variables returned from `extractNamedConstraints`
#     # are correctly sorted by the time index!
#     data = [
#         c.getAttr("Slack")
#         for (n, c) in extractNamedConstraints(model).items()
#         if n.startswith(name + "[") or n == name
#     ]
#     return np.array(data)


# def objectiveValue(model):
#     return(model.getObjective().getValue())
