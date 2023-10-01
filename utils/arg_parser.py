from argparse import ArgumentParser
def parse_arguments():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, default='FCDN')
    arg_parser.add_argument('--dataset_name', type=str, default='Transistor_dataset')
    arg_parser.add_argument('--device', type=str, default='cuda')
    args = arg_parser.parse_args()
    print("Argument values:")
    print('model_name: ', args.model_name)
    print('dataset_name: ',args.dataset_name)
    print('device: ',args.device)
    return args

