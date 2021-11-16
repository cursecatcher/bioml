#!/usr/bin/env python3 

import argparse
from dataset import Dataset
from os.path import basename


outfile_suffix = "_population.tsv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", type=str, nargs="*", required=False)
    parser.add_argument("-c", "--counts", type=str, nargs="*", required=False)
    parser.add_argument("-n", "--names", type=str, nargs="*", required=False)
    parser.add_argument("-p", "--populations", type=str, nargs="*", required=False)
    args = parser.parse_args() 

    populations = args.populations
    samples, countsmatrices, sample_names = args.samples, args.counts, args.names
    if len(countsmatrices) == 1:
        #epik trick to sample from a giant count matrix 
        countsmatrices = [countsmatrices] * len(samples)
    

    if populations:
        name = "-".join([
            basename(p).replace(outfile_suffix, "") for p in populations])
        datasets = [Dataset(p) for p in populations]
        new = datasets[0].merge(datasets[1:])
        new.data.to_csv(f"{name}{outfile_suffix}", sep="\t")

    else:
        assert samples and countsmatrices and sample_names
        assert len(samples) == len(countsmatrices) == len(sample_names)
        
        datasets = list()

        for sample, countmatrix, name in zip(samples, countsmatrices, sample_names):
            # print(sample, countmatrix, name)
            d = Dataset(sample)
            d.load_data(countmatrix)
            d.data.to_csv(f"{name}{outfile_suffix}", sep="\t")
            datasets.append( d )
