#!/usr/bin/env python3 

import argparse
from dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", type=str, nargs="+", required=True)
    parser.add_argument("-c", "--counts", type=str, required=True)
    parser.add_argument("-n", "--names", type=str, nargs="+", required=True)
    args = parser.parse_args() 

    samples, countsmatrices, sample_names = args.samples, args.counts, args.names
    countsmatrices = [countsmatrices] * len(samples)

    assert len(samples) == len(countsmatrices) == len(sample_names)
    
    datasets = list()

    for sample, countmatrix, name in zip(samples, countsmatrices, sample_names):
        d = Dataset(sample)
        d.load_data(countmatrix)
        d.data.to_csv(f"{name}_population.tsv", sep="\t")
        datasets.append( d )


