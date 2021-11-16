#!/usr/bin/env python3 

import argparse
from collections import defaultdict
import logging
import os, subprocess, sys
import shutil
import time 




def get_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    ####################### IO 
    #parent output folder where to save all results
    parser.add_argument("-o", "--outfolder", type=str, required=False)       #IGNORED - here just to catch that argument
    parser.add_argument("-i", "--input_data", type=str, required=True)      #input dataset 
    parser.add_argument("-m", "--more_data", type=str, required=False)      #additional data to integrate in the input dataset
    parser.add_argument("-f", "--feature_lists", type=str, nargs="+")       #list of feature lists 
    parser.add_argument("-v", "--validation_sets", type=str, nargs="*")     #list of validation sets
    # ###################### PREDICTION 
    # #target covariate to predict - only categorical features 
    # parser.add_argument("-t", "--target", type=str, required=True)          #name of the (categorical) feature to be predicted 
    # parser.add_argument("-l", "--labels", type=str, nargs=2)                #pair of labels (neg label, pos label)
    # parser.add_argument("-p", "--pos_labels", type=str, nargs="+")          #labels to be considered as positive     
    # parser.add_argument("-n", "--neg_labels", type=str, nargs="+")          #labels to be considered as negative
    # ###################### 
    # parser.add_argument("--trials", type=int, default=1)                    #num of runs to be done 
    # parser.add_argument("--ncv", type=int, default=10)                      #number of folds to be used during cross validation 
    # specific for classification task 
    # parser.add_argument("--vsize", type=float, default=0.1)

    return parser 



def check_for_existence(files: list):
    return { f: os.path.exists(f) for f in files }

def format_args(args, io_args: dict) -> list:
    def flat_strlist(l: list):
        return " ".join( l )

    argstr = list()

    ####################### non-IO stuff 
    # argstr.append( f"--target {args.target} --trials {args.trials} --ncv {args.ncv}" )
    # if args.labels:
    #     argstr.append( f"--labels {flat_strlist(args.labels)}" )

    # elif all( [args.pos_labels, args.neg_labels] ):
    #     argstr.append( f"--pos_labels {flat_strlist(args.pos_labels)}" )
    #     argstr.append( f"--neg_labels {flat_strlist(args.neg_labels)}" )
    # else:
    #     sys.exit("Some information about labels is missing")
    

    ####################### IO stuff 
    actual_input = io_args.pop("input_data")
    argstr.append( f"--input_data {actual_input[0]}")
    if len(actual_input) == 2:                          #get additional data, if provided 
        argstr.append( f"--more_data {actual_input[1]}")

    for cat, files in io_args.items():
        if len(files):  #feature lists and independent validation sets
            argstr.append( f"--{cat} {flat_strlist(files)}")
        
    return argstr
    


if __name__ == "__main__":
    parser = get_parser("bioml")
    parser.add_argument("-d", "--docker_outfolder", type=str, required=True)
    parser.add_argument("--container_name", type=str, required=False)

    parser.add_argument("--clf", action="store_true")
    parser.add_argument("--fsel", action="store_true")
    parser.add_argument("--rm", action="store_true")
    
    args, unknownargs = parser.parse_known_args()
    operations = [args.clf, args.fsel]

    input_files = dict(
        input_data = [ str(args.input_data) ],
        feature_lists = list(args.feature_lists),
        validation_sets = list(args.validation_sets)
    )
    if args.more_data:
        input_files["input_data"].append( str(args.more_data) )

    my_files = dict()

    for cat, files in input_files.items():
        my_files.update( check_for_existence(files) )
        
    if not all(my_files.values()):
        wrong_files = [f for f, b  in my_files.items() if not b]
        sys.exit(f"ERROR: some input file has not been found:\n{wrong_files}")


    too_much, not_enough = all(operations), not any(operations)

    if too_much:
        print("Just one operation at the time", file=sys.stderr)
    elif not_enough:
        print("No operation selected", file=sys.stderr)

    if too_much or not_enough:
        sys.exit("""Please choose an operation:
- classification: --clf
- feature selection: --fsel""")
    

    #parse args, create docker outfolder
    docker_outfolder = args.docker_outfolder 
    new_files_collection = defaultdict(list)

    if not os.path.exists(docker_outfolder):
        os.mkdir(docker_outfolder)                                      #build folder for docker container 
        docker_outfolder = os.path.abspath(docker_outfolder)            #get abspath for mounting 
        print(f"Outfolder created in {docker_outfolder}")
        try:
            for cat, cat_files in input_files.items():
                outpath = os.path.join(docker_outfolder, cat)           #build a subfolder for each file category
                os.mkdir(outpath)
                dockpath = os.path.join("/data", cat)                   #get subfolder path in the container 

                for f in cat_files:
                    basename_f = os.path.basename(f)

                    shutil.copyfile(f, os.path.join(outpath, basename_f)) #copy file in the docker folder
                    new_files_collection[cat].append( 
                        os.path.join(dockpath, basename_f))             #add input file with new path 

        except Exception as e:
            shutil.rmtree(docker_outfolder)    
            raise Exception(e)
    else:
        sys.exit("Please provide a new docker outfolder")
        

    docker_run = list() 

    formatted_args = format_args(args, new_files_collection)
    cidfile = os.path.join(docker_outfolder, 'dockerID')
    script = "feature_selection.py" if args.fsel else "classification.py"
    

    docker_run.append("docker run -d --user 1000:1000")
    docker_run.append(f"-v {docker_outfolder}:/data")
    docker_run.append(f"--cidfile {cidfile}")

    if args.container_name:
        docker_run.append(f"--name {args.container_name}")
    docker_run.append( f"cursecatcher/bioml {script}" )
    docker_run.append( " ".join(formatted_args) )
    docker_run.append( "-o /data/results")
    # if args.clf:
    #     docker_run.append(f"--vsize {args.vsize}")
    if unknownargs:
        logging.info(f"Adding the following additional parameters: {unknownargs}")
        #TODO - remove -o parameter if it has been passed 
        docker_run.extend( unknownargs )


    docker_command = " ".join(docker_run)
    print(f"Running the following container:\n{docker_command}")

    sp = subprocess.run(docker_command, shell=True)
    
    while not os.path.exists(cidfile):
        time.sleep(1)
    
    with open(cidfile) as f:
        cid = f.readline().strip()

    print("Showing current execution. Press ctrl+c to quit.")
    try:
        subprocess.run(f"docker logs -f {cid}", shell=True)
    except KeyboardInterrupt:
        print(f"Ok, bye.\nPs. your container ID is {cid}")


