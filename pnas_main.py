import argparse
import pnas

"""
Main file to run both random search and PNAS
"""

config ={}

if __name__=="__main__":

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='PNAS')
    parser.add_argument("--dir_path", type=str, default="pnas/temp", help='')
    parser.add_argument("--seed", type=int, default=10, help='Set the seed for reproducibility')
    parser.add_argument("--pnas", default=False, action='store_true', help='')

    #Arguments cnn
    parser.add_argument("--num_blocks", type=int, default=3, help='Number of Blocks')
    parser.add_argument("--num_repetitions", type=int, default=2, help='Number of repetitions of stride 1 cells')
    parser.add_argument("--num_filters", type=int, default=24, help='Number of Filters')
    parser.add_argument("--k", type=int, default=64, help='Number of models for the next step')
    parser.add_argument("--batch_size_cifar", type=int, default=512, help='Batch size cifar')

    parser.add_argument("--ensemble", default=False, action='store_true', help='')

    args = parser.parse_args()

    config['dir_path']=args.dir_path
    config['seed'] = args.seed
    config['pnas'] = args.pnas

    print(args.pnas)

    config['num_blocks'] = args.num_blocks
    config['num_repetitions'] = args.num_repetitions
    config['num_filters'] = args.num_filters
    config['k'] = args.k
    config['batch_size_cifar'] = args.batch_size_cifar

    config['ensemble'] = args.ensemble

    pnas.run(config)