import sys
import argparse
import numpy as np

# Read run index
testno = sys.argv[1]
outfile = "DiscConfig{}.json".format(testno)

# Define input and output files
default_config = open("../KeyScripts/DiscConfig_default.json",'r')
new_config = open(outfile,'w')

# Extract defaults from DiscConfig_default.json
defaults = default_config.readlines()

words = []
for i in range(0,len(defaults),1):
	words.append(defaults[i].split())

"""
# Customise the directly set parameters
"""
#M_star = float(sys.argv[2])			# In solar masses
#M_disc = float(sys.argv[2])			# As a fraction of the stellar mass
mass_loss = sys.argv[5]
UV = float(sys.argv[4])
#R1 = float(sys.argv[5])
Rc = float(sys.argv[2])
alpha = float(sys.argv[3])
#words[4][2] = str(R1)+","
words[9][2] = str(alpha)+","
#words[10][2] = str(M_disc)+","
words[11][2] = str(Rc)+","
#words[25][2] = str(M_star)+","
#words[37][2] = words[37][2][:-1]+ '_' + str(testno) + '"'	# Update output folder name
#words[45][2] = words[45][2][:-1]+ '_' + str(testno) + '"'	# Update plot filename
words[50][2] = str(UV)
words[49][2] = '"' + mass_loss + '",' # Set type of mass loss from {FRIED, compare, Integrated, Constant}

"""
# Update for varying timestep
words[37][2] = words[37][2][:-2]+ '_' + str(testno) + '",'	# Update output folder name
words[45][2] = words[45][2][:-1]+ '_' + str(testno) + '"'	# Update plot filename
words[42][2] = sys.argv[2]+","
"""

"""
step = int(np.fromstring(words[42][2],dtype=float,sep=','))
end = int(np.fromstring(words[41][2],dtype=float,sep=','))
number = end/step
"""

words[44][2] = np.array2string(np.append([0],np.logspace(3,7,41,endpoint=True,base=10,dtype=int)),max_line_width=1e4,separator=',') + ',' # Log times
words[41][2] = str(1e7) + ","
words[37][2] = words[37][2][:-2]+ '_' + str(testno) + '",'	# Update output folder name
words[45][2] = words[45][2][:-1]+ '_' + str(testno) + '"'	# Update plot filename

# Write the output json file and close
for i in range(0,len(defaults),1):
	output = " ".join(words[i])
	new_config.write(output)
	new_config.write("\n")

default_config.close()
new_config.close()
