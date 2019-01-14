from run_model import timeplot

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default=DefaultModel)
args = parser.parse_args()
model = json.load(open(args.model, 'r'))

inputdata = np.loadtxt
