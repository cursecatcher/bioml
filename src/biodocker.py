#!/usr/bin/env python3 

import argparse, getpass, time 
from collections import defaultdict
import logging
import os, subprocess, sys, shutil

logging.basicConfig(level=logging.DEBUG)

INPUT_DATA = "input_data"
MORE_DATA = "more_data"
FEATURE_LISTS = "feature_lists"
VALIDATION_SETS = "validation_sets"
VALID_SAMPLES_ID = "vid"



def get_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    ####################### IO 
    #parent output folder where to save all results
    parser.add_argument("-o", "--outfolder", type=str, required=False)         #IGNORED - it's here just to catch that argument
    parser.add_argument("-i", f"--{INPUT_DATA}", type=str, required=True)      #input dataset 
    parser.add_argument("-m", f"--{MORE_DATA}", type=str, required=False)      #additional data to integrate in the input dataset
    parser.add_argument("-f", f"--{FEATURE_LISTS}", type=str, nargs="+")       #list of feature lists 
    parser.add_argument("-v", f"--{VALIDATION_SETS}", type=str, nargs="*")     #list of validation sets

    parser.add_argument(f"--{VALID_SAMPLES_ID}", type=str, required=False)     #list of samples to be used as validation set 

    parser.add_argument("--as_root", action="store_true")

    return parser 



def check_for_existence(files: list):
    return { f: os.path.exists(f) for f in files }

def format_args(io_args: dict) -> list:
    argstr = list()    

    actual_input = io_args.pop(INPUT_DATA)
    argstr.append( f"--{INPUT_DATA} {actual_input[0]}")
    if len(actual_input) == 2:                          #get additional data, if provided 
        argstr.append( f"--{MORE_DATA} {actual_input[1]}")

    #add feature lists and independent validation sets
    argstr.extend([ 
        f"--{cat} {' '.join(files)}"
            for cat, files in io_args.items() 
                if len(files) ])

    return argstr
    


if __name__ == "__main__":
    parser = get_parser("bioml")
    parser.add_argument("-d", "--docker_outfolder", type=str, required=True)
    parser.add_argument("--container_name", type=str, required=False)

    parser.add_argument("--clf", action="store_true")
    parser.add_argument("--fsel", action="store_true")
    parser.add_argument("--rm", action="store_true")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tag", type=str, default="latest")
    
    args, unknownargs = parser.parse_known_args()
    operations = [args.clf, args.fsel]

    input_files = dict() 
    #training set 
    input_files[INPUT_DATA] = [ str(args.input_data) ]
    if args.more_data:
        input_files[INPUT_DATA].append( str(args.more_data) )

    #feature lists & validation sets 
    input_files[FEATURE_LISTS] = list(args.feature_lists)
    if args.validation_sets:
        input_files[VALIDATION_SETS] = list(args.validation_sets)
    if args.vid:
        #optional parameter for classification
        input_files[VALID_SAMPLES_ID] = [ args.vid ]


    my_files = dict()

    for cat, files in input_files.items():
        my_files.update( check_for_existence(files) )
        
    if not all(my_files.values()):
        wrong_files = [f for f, b  in my_files.items() if not b]
        sys.exit(f"ERROR: some input file has not been found:\n{wrong_files}")


    too_much, not_enough = all(operations), not any(operations)

    if too_much or not_enough:
        if too_much:
            print("Just one operation at the time", file=sys.stderr)
        else:
            print("No operation selected", file=sys.stderr)

        sys.exit("Please select an operation:\n- classification: --clf\n- feature selection: --fsel")
    

    #parse args, create docker outfolder
    docker_outfolder = os.path.abspath( args.docker_outfolder )
    new_files_collection = defaultdict(list)

    if not os.path.exists(docker_outfolder):
        os.mkdir(docker_outfolder)                                      #build folder for docker container 
        print(f"Outfolder created in {docker_outfolder}")
        try:
            for cat, cat_files in input_files.items():
                outpath = os.path.join(docker_outfolder, cat)           #build a subfolder for each file category
                os.mkdir(outpath)
                dockpath = os.path.join("/data", cat)                   #get subfolder path in the container 

                for f in cat_files:
                    if os.path.isfile(f):
                        basename_f = os.path.basename(f)
                        #copy file and attach it to the new name to the proper list 
                        shutil.copyfile(f, os.path.join(outpath, basename_f)) #copy file in the docker folder
                        new_files_collection[cat].append( 
                            os.path.join(dockpath, basename_f))             #add input file with new path 
                    elif os.path.isdir(f):
                        new_dir_name = os.path.split( f.rstrip("/") )[-1]
                        new_dir = os.path.join(outpath, new_dir_name)

                        shutil.copytree(f, os.path.join(outpath, new_dir))

                        for root, folders, copied_files in os.walk(new_dir):
                            new_files_collection[cat].extend([
                                os.path.join(dockpath, new_dir_name, new_file)
                                    for new_file in copied_files
                            ])

        except Exception as e:
            shutil.rmtree(docker_outfolder)    
            raise Exception(e)
    else:
        sys.exit("Please provide a new docker outfolder")
        
    docker_run = list() 

    formatted_args = format_args(new_files_collection)
    cidfile = os.path.join(docker_outfolder, 'dockerID')
    script = "feature_selection.py" if args.fsel else "classification.py"

    docker_run.append(f"docker run -d --cidfile {cidfile} -v {docker_outfolder}:/data")
    
    if not args.as_root:
        get_id = f"$(id -u {getpass.getuser()})"
        as_user = f"-u {get_id}:{get_id}"
        docker_run.append( as_user )

    if args.container_name:
        docker_run.append(f"--name {args.container_name}")
    
    image_tag = f":{args.tag}"
    docker_run.append( f"cursecatcher/bioml{image_tag} {script} -o /data/{os.path.basename(args.docker_outfolder)}_results" ) #set output folder 
    docker_run.append( " ".join(formatted_args) )
    
    if unknownargs:
        logging.info(f"Adding the following additional parameters: {unknownargs}") 
        docker_run.extend( unknownargs )


    docker_command = " ".join(docker_run)    
    with open(os.path.join(docker_outfolder, "COMMAND"), "w") as f:
        f.write(docker_command)

    print(f"Running the docker container:\n{docker_command}\n")

    sp = subprocess.run(docker_command, shell=True)
    
    while not os.path.exists(cidfile):
        time.sleep(1)
    
    with open(cidfile) as f:
        cid = f.readline().strip()

    if args.verbose:
        print("Showing current execution. Press ctrl+c to quit.")
        try:
            subprocess.run(f"docker logs -f {cid}", shell=True)
        except KeyboardInterrupt:
            print(f"\nOk, bye.\nPs. your container ID is {cid}")


