from run_model import timeplot
import numpy as np

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="_default")
args = parser.parse_args()
loadfile = "DiscConfig"+args.model+".json"
model = json.load(open(loadfile, 'r'))

inputdata = np.loadtxt(model['output']['directory']+"/"+model['output']['plot_name']+"_discproperties.dat")

timeplot(model,inputdata,None)
