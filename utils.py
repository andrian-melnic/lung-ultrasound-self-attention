from tabulate import tabulate
import os

def get_model_name(args):
    # -Logger configuration
    version = f"V{args.version}" if args.version else "V1"
    version = version.strip()

    version = f"V{args.version}" if args.version else "V1"

    name_version = f"_{version}"
    name_trained = "_pretrained" if args.pretrained==True else ""
    name_layer = f"_{args.freeze_layers}" if args.freeze_layers else ""

    model_name = f"{args.model}{name_version}{name_trained}{name_layer}/{args.optimizer}/ds_{args.ratios[0]}_lr{args.lr}_bs{args.batch_size}"
    return model_name, version

def generate_table(name, data, exclude=[]):
    table_data = []
    table_data.append([name])
    for key, value in data.items():
        if key not in exclude:
            table_data.append([key, value])

    table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
    print("\n\n" + table)
    
def check_checkpoint(chkp):
    print("Checkpoint mode activated...\n")
    if (chkp == "best"):
        print("Loading BEST checkpoint...\n")
    if (chkp == "last"):
        print("Loading LAST checkpoint...\n")
    else:
    # Check if checkpoint file exists
        if not os.path.isfile(chkp):
            print(f"Checkpoint file '{chkp}' does not exist. Exiting...")
            exit()
    print(f"Loading checkpoint from PATH: '{chkp}'...\n")