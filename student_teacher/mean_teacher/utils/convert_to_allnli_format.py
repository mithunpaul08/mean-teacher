import pandas as pd
import os



def read_fever_format_write_to_allnli_format(input_file, run_name, output_folder):
    """Load dataset

    Args:
        args (str): all arguments which were create initially.
    Returns:
        an instance of ReviewDataset
    """

    claim_ev_lex= pd.read_json(input_file, lines=True)

    s1_filename = os.path.join(output_folder, "s1." + run_name)
    with open(s1_filename, 'w') as outfile:
        for claim in (claim_ev_lex.claim):
            outfile.write(claim + "\n")


    s2_filename = os.path.join(output_folder, "s2." + run_name)
    with open(s2_filename, 'w') as outfile:
        for ev in (claim_ev_lex.evidence):
            outfile.write(ev+"\n")

    labels_filename = os.path.join(output_folder, "labels." + run_name)
    with open(labels_filename, 'w') as outfile:
        for label in (claim_ev_lex.label):
            outfile.write(label.lower()+"\n")
abs=os.path.abspath(os.path.dirname(__file__))
os.chdir(abs)
path_base= "../../data/rte/"

#convert from fever format. this should be run only once ever ideally-per every type of file.
input_file=(os.path.join(path_base,"fever/train/fever_train_lex.jsonl"))
output_folder=(os.path.join(path_base, "fever/allnli/lex/"))
read_fever_format_write_to_allnli_format(input_file,"train",output_folder)

input_file=(os.path.join(path_base,"fever/dev/fever_dev_lex.jsonl"))
read_fever_format_write_to_allnli_format(input_file,"dev",output_folder)

output_folder=(os.path.join(path_base, "fnc/allnli/lex/"))
input_file=(os.path.join(path_base,"fnc/dev/fnc_dev_lex.jsonl"))
read_fever_format_write_to_allnli_format(input_file,"dev",output_folder)

###---all delexicalized files
input_file=(os.path.join(path_base, "fever/train/fever_train_delex.jsonl"))
output_folder=(os.path.join(path_base, "fever/allnli/delex/"))
read_fever_format_write_to_allnli_format(input_file,"train",output_folder)
input_file=(os.path.join(path_base, "fever/dev/fever_dev_delex.jsonl"))
read_fever_format_write_to_allnli_format(input_file,"dev",output_folder)

output_folder=(os.path.join(path_base, "fnc/allnli/delex/"))
input_file=(os.path.join(path_base, "fnc/dev/fnc_dev_delex.jsonl"))
read_fever_format_write_to_allnli_format(input_file,"dev",output_folder)
