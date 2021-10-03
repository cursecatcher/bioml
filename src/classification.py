#!/usr/bin/env python3 

import argparse
import dataset as ds 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="filename", type=str, required=True)
    parser.add_argument("-m", "--more_input", dest="more_input", type=str, required=False)
    parser.add_argument("-f", "--features", type=str)

    args = parser.parse_args()

    d = ds.Dataset(args.filename)
    if args.more_input:
        d.load_data(args.more_input)

    if args.features:
        print("Extracting features?")
        flist = ds.FeatureList(args.features)

        b = ds.BinaryClfDataset(d, "Class", ["CRC","Healthy"])
        c = b.extract_subdata(flist)

    

        





