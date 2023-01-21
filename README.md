# BLI

A set of methods for fitting BLI sensorgram to kinetic models (1:1, bivalent analyte, heterogeneous ligand, two-state).
LMFIT is used to determine kinetic parameters. Both MCK and SCK are supported. Preprocessing tools, plotting and statistical analysis are included.

Quick start
from experiment import *
from models import *
exp = Experiment()

#import data
exp.load_bli_data(EXPORTED_BLI_CSV_FILES) 

#assign analyte concentrations for each dataset
cs = [0.1, 0.2, 0.4, 0.8, 1.6]
cs = [c*1e-6 for c in cs]
for i, ds in enumerate(exp):
  ds.steps[0].concentration = cs[i] #first step is association with analyte concentration >0
 
#preprocess and plot data
exp.crop(2, inplace=True)
exp.interstep_correction()
exp.plot()

#assign model and create params
exp.model = models.one_to_one()
exp.create_params(offsets=False, mtl=False)

#fit data to model
exp.fit_params()

#plot data, show fitted curve
exp.plot(fit=True, correct_offsets=False)

#print fitted parameter values
exp.params
