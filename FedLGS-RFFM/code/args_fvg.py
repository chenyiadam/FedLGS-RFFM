
import argparse
import torch


def args_parser_live():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--E', type=int, default=15, help='number of rounds of training')   
    parser.add_argument('--r', type=int, default=10, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=20, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')
    parser.add_argument('--B', type=int, default=30, help='local batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--epcluo', type=float, default=0.1, help='epcluo')
    parser.add_argument('--pr', type=float, default=0.5, help='pr')
    parser.add_argument('--batch_sample_rate', type=float, default=0.5, help='pr')


    clients = ['Task' + str(i) for i in range(0, 20)]
    parser.add_argument('--clients', default=clients)

    # args = parser.parse_args()
    # args,unknow = parser.parse_known_args()
    
    args = parser.parse_known_args()[0]
    return args




def args_parser_shake():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--E', type=int, default= 3, help='number of rounds of training')   
    parser.add_argument('--r', type=int, default= 3, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=20, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')
    parser.add_argument('--B', type=int, default=5, help='local batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--epcluo', type=float, default=0.1, help='epcluo')
    parser.add_argument('--pr', type=float, default=0.5, help='pr')
    parser.add_argument('--batch_sample_rate', type=float, default=0.5, help='pr')

    
    
    clients = ['Task' + str(i) for i in range(0, 20)]
    parser.add_argument('--clients', default=clients)

    # args = parser.parse_args()
    # args,unknow = parser.parse_known_args()
    
    args = parser.parse_known_args()[0]
    return args