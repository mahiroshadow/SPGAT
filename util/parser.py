import argparse

parser=argparse.ArgumentParser(description="Run SPGAT.")
parser.add_argument("--dim", type=int, default=64,help="embedding len")
parser.add_argument("--epoches",type=int,default=100)
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--weight_decay",type=float,default=5e-4)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--p",type=float,default=0.2)
arg=parser.parse_args()

