import run_model
import numpy as np

import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="_default")
args = parser.parse_args()
loadfile = "DiscConfig"+args.model+".json"
model = json.load(open(loadfile, 'r'))

#load_ref = "DiscConfig"+str(6)+".json"
#model_ref = json.load(open(load_ref, 'r'))

inputdata = np.loadtxt(model['output']['directory']+"/"+model['output']['plot_name']+"_discproperties.dat")
#input_ref = np.loadtxt(model_ref['output']['directory']+"/"+model_ref['output']['plot_name']+"_discproperties.dat")

disc, _, _, _, _, _, Dt_nv = run_model.setup_wrapper(model)
run_model.timeplot(model, inputdata, np.column_stack((Dt_nv/(2*np.pi), disc.grid.Rc)), data_2 = None, logtest=False)
