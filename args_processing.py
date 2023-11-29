from argparse import ArgumentParser

import json

# ------------------------------ Parse arguments ----------------------------- #
def parse_arguments():
    # Parse command-line arguments
    parser = ArgumentParser()

    allowed_models = ["google_vit", 
                    "resnet18",
                    "resnet10",
                    "resnet50",
                    "beit", 
                    'timm_bot', 
                    "botnet18", 
                    "botnet50",
                    "vit",
                    "swin_vit",
                    "simple_vit"]

    allowed_modes = ["train", "test", "train_test"]
    parser.add_argument("--model", type=str, choices=allowed_models)
    parser.add_argument("--mode", type=str, choices=allowed_modes)
    parser.add_argument("--version", type=str)
    parser.add_argument("--working_dir_path", type=str)
    parser.add_argument("--dataset_h5_path", type=str)
    parser.add_argument("--hospitaldict_path", type=str)
    
    parser.add_argument("--trim_data", type=float)
    parser.add_argument("--trim_train", type=float)
    parser.add_argument("--trim_test", type=float)
    parser.add_argument("--trim_val", type=float)
    
    parser.add_argument('--ratios', nargs='+', type=float, help='Sets ratios')


    parser.add_argument("--chkp", type=str)
    parser.add_argument("--rseed", type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.001)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--precision", default=32)
    parser.add_argument("--disable_warnings", dest="disable_warnings", action='store_true')
    parser.add_argument("--pretrained", dest="pretrained", action='store_true')
    parser.add_argument("--freeze_layers", type=str)
    parser.add_argument("--test", dest="test", action='store_true')
    parser.add_argument("--mixup", dest="mixup", action='store_true')
    parser.add_argument("--augmentation", dest="augmentation", action='store_true')
    parser.add_argument("--summary", dest="summary", action='store_true')
    

    # Add an argument for the configuration file
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')

    args = parser.parse_args()

    # -------------------------------- json config ------------------------------- #

    config_path = 'configs/configs.json'
    selected_config = None
    # If a configuration file was provided, load it
    if args.config:
        with open(config_path, 'r') as f:
            configurations = json.load(f)
        for config in configurations:
            if config['config'] == args.config:
                selected_config = config
                break

        # Override the command-line arguments with the configuration file
        for key, value in selected_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
                
        # Check and set the ratios
        if "ratios" in selected_config:
            ratios = selected_config["ratios"]
            if len(ratios) != 3 or sum(ratios) != 1:
                parser.error('Invalid ratios provided in the configuration file')
            
        
    print(f"args are: {args}")

    return args