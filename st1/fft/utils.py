#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:48:52 2023

@author: lesya
"""

from sklearn.model_selection import train_test_split

def upsample(lang, train_df, satire_df):     
    # upsamples the data for "reporting" and "opinion" classes from the same dataset 
    # for "satire" class, upsamples it from the external dataset
    data_upsampled=train_df[train_df["language"]==lang]
    if lang!="en":
        while True:
            for i, row in data_upsampled[data_upsampled["label"]=="satire"].iterrows():
                if len(data_upsampled[data_upsampled["label"]=="satire"])<len(data_upsampled[data_upsampled["label"]=="opinion"]):
                    data_upsampled=data_upsampled.append(row)
            if len(data_upsampled[data_upsampled["label"]=="satire"])>=len(data_upsampled[data_upsampled["label"]=="opinion"]):
                break
    else:
        for i, row in satire_df.iterrows():
            data_upsampled=data_upsampled.append(row)
     
    while True:
        for i, row in data_upsampled[data_upsampled["label"]=="reporting"].iterrows():
            if len(data_upsampled[data_upsampled["label"]=="reporting"])<len(data_upsampled[data_upsampled["label"]=="opinion"]):
                data_upsampled=data_upsampled.append(row)
        if len(data_upsampled[data_upsampled["label"]=="reporting"])>=len(data_upsampled[data_upsampled["label"]=="opinion"]):
            break
    return data_upsampled




def split_stratified_into_3_folds(df_input, stratify_colname='y',
                                         frac_train=0.33, frac_val=0.33, frac_test=0.34,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of the columns on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test
