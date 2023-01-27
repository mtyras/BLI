from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np
from copy import copy

# The mass transfer coefficient can be normalized for molecular weight and adjusted 
# approximately for the conversion of surface concentration to RU, to give a parameter 
# referred to as the mass transfer constant kt
#  (units RU·M-1 · s-1):
# where G is the conversion factor from surface concentration to RU. The value of G is 
# approximately 10^9
#  for proteins on Sensor Chip CM5

import sys, inspect
def get_models_dict():
  models_dict = {}
  for name, obj in inspect.getmembers(sys.modules['models']):
      if inspect.isclass(obj) and obj.__module__ == 'models':
          models_dict[obj.name] = obj
  return models_dict



class One_to_one:
  name = 'One to one'
  latex = r'''
              A_{bulk} \underset{kt}{\stackrel{kt}{\rightleftharpoons}} A_{surf} + L \underset{kd}{\stackrel{ka}{\rightleftharpoons}} LA
              \\[0.2in]
              \frac{d[A_{surf}]}{dt} = k_t([A_{bulk}] - [A_{surf}]) - (k_a [L] [A_{surf}] - k_d [LA])
              \\[0.2in]
              \frac{d[L]}{dt} = - (k_a[L][A_{surf}]-k_d[LA])
              \\[0.2in]
              \frac{d[LA]}{dt}=k_a[L][A_{surf}]-k_d[LA]
          '''
  no_ODEs = 1 #excluding mtl
  signal_components = 1 #components that contribute to response

  def __init__(self):
    self.params_definitions = {
      'ka' : Parameter(name = 'ka', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd' : Parameter(name = 'kd', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax' : Parameter(name = 'ymax', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = 0, min = -1000, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }

  def __repr__(self):
    return self.name
  
  def ydot(self, t, y, params, c0, ds_index):
    #unpack params
    ka = params['ka'].value
    kd = params['kd'].value
    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    
    ymax = params[f'ymax_ds{ds_index}'].value
    Abulk = c0

    #ODE system
    if include_mtl:
      [LA, Asurf] = y
      L = ymax - LA
      dAsurf = kt*(Abulk - Asurf) - (ka*L*Asurf - kd*LA)
      dLA = ka*L*Asurf - kd*LA
      return [dLA, dAsurf]
    
    else:
      [LA] = y
      L = ymax - LA
      dLA = ka*L*Abulk - kd*LA
      return [dLA]


class Bivalent_analyte:
# Once analyte is attached to the ligand through binding at the first site, interaction at the second site does not
# contribute to the SPR response. For this reason, the association rate constant for the second interaction is reported
# in units of RU-1s-1, and can only be obtained in M-1s-1 if a reliable conversion factor between RU and M is available.
# For the same reason, a value for the overall affinity or avidity constant is NOT reported.

#Assuming that 1 RU = 1 pg/mm2 and that the flexible dextran matrix on the sensor surface extends 100 
# nm into solution (Stenberg et al., 1991) the response may be interpreted in concentration terms 
# and 1RU = 10 ug/ml. By introducing the molecular  weight (MW) of the analyte numerical values for 
# rate constants can be compared and: ka2 = ka2 * 100 * MW(A) (Karlsson 1995)
# ka2 (M-1s-1) = ka2 (RU-1 s-1) * Mr * 100 via Karlsson, R., J. A. Mo and R. Holmdahl, JImmMeth 1995

  name = 'Bivalent analyte'
  latex = r'''
          A_{bulk} \underset{kt}{\stackrel{kt}{\rightleftharpoons}} A_{surf}
          \\[0.2in]
          2L + A_{surf} \underset{kd1}{\stackrel{ka1}{\rightleftharpoons}} LA + L \underset{kd2}{\stackrel{ka2}{\rightleftharpoons}} LLA
          '''
  no_ODEs = 2
  signal_components = 2

  def __init__(self) -> None:
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+08, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 'RU-1s-1'}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax' : Parameter(name = 'ymax', vary = True, value = 1, min = 0, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = 0, min = -1000, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }
  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ymax = params[f'ymax_ds{ds_index}'].value
    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    Abulk = c0

    #ODE system
    if include_mtl:
      [LA, LLA, Asurf] = y
      L = ymax - LA - LLA
      dAsurf = kt*(Abulk - Asurf) - (2*ka1*L*Asurf - kd1*LA)
      dLA = 2*ka1*L*Asurf - kd1*LA - (ka2*LA*L - 2*kd2*LLA)
      dLLA = ka2*LA*L - 2*kd2*LLA
      return [dLA, dLLA, dAsurf]
    
    else:
      [LA, LLA] = y
      L = ymax - LA - LLA
      dLA = 2*ka1*L*Abulk - kd1*LA - (ka2*LA*L - 2*kd2*LLA)
      dLLA = ka2*LA*L -2*kd2*LLA
      return [dLA, dLLA]



class Two_state:
  name = 'Two state'
  latex = r'''
          L + A \underset{kd1}{\stackrel{ka1}{\rightleftharpoons}} LA
          \\[0.2in]
          LA \underset{kd2}{\stackrel{ka2}{\rightleftharpoons}} LA*
          '''
  no_ODEs = 2
  signal_components = 2

  def __init__(self) -> None:
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax' : Parameter(name = 'ymax', vary = True, value = 1, min = 0, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = 0, min = -1000, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }
  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ymax = params[f'ymax_ds{ds_index}'].value
    
    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    Abulk = c0

    #ODE system
    if include_mtl:
      [LA, LS, Asurf] = y
      L = ymax - LA - LS
      
      dAsurf = kt*(Abulk - Asurf) - (ka1*L*Asurf - kd1*LA)   
      dLA = ka1*L*Asurf - kd1*LA - (ka2*LA - kd2*LS)
      dLS = ka2*LA - kd2*LS
      
      return [dLA, dLS, dAsurf]
    
    else:
      [LA, LS] = y
      L = ymax - LA - LS
      
      dLA = ka1*L*Abulk - kd1*LA - (ka2*LA - kd2*LS)
      dLS = ka2*LA - kd2*LS
      
      return [dLA, dLS]

#TO DO!!!!!!!!
#needs an expression to bind ymax1 with ymax2 with a constant
class Heterogeneous_ligand:
  name = 'Heterogeneous ligand'
  latex = r'''
          L1 + A \underset{kd1}{\stackrel{ka1}{\rightleftharpoons}} L1A
          \\[0.2in]
          L2 + A \underset{kd1}{\stackrel{ka1}{\rightleftharpoons}} L2A
          '''
  no_ODEs = 2
  signal_components = 2

  def __init__(self) -> None:
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e+04, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax1' : Parameter(name = 'ymax1', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'ymax2' : Parameter(name = 'ymax2', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = -1000, min = 0, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }

  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ymax1 = params[f'ymax1_ds{ds_index}'].value
    ymax2 = params[f'ymax2_ds{ds_index}'].value
    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    Abulk = c0

    #ODE system
    if include_mtl:
      [L1A, L2A, Asurf] = y
      L1 = ymax1 - L1A
      L2 = ymax2 - L2A
      
      dAsurf = kt*(Abulk - Asurf) - (ka1*L1*Asurf - kd1*L1A) - (ka2*L2*Asurf - kd2*L2A)   
      dL1A = ka1*L1*Asurf - kd1*L1A
      dL2A = ka2*L2*Asurf - kd2*L2A
      
      return [dL1A, dL2A, dAsurf]
    
    else:
      [L1A, L2A] = y
      L1 = ymax1 - L1A
      L2 = ymax2 - L2A
      
      dL1A = ka1*L1*Abulk - kd1*L1A
      dL2A = ka2*L2*Abulk - kd2*L2A
      
      return [dL1A, dL2A]


class Bivalent_ligand:
  name = 'One to two'
  latex = r'''
  L + A = LA
  LA + A = LAA
  '''
  no_ODEs = 2
  signal_components = 2

  def __init__(self) -> None:
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e+04, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax' : Parameter(name = 'ymax', vary = True, value = 1, min = 0, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = -1000, min = 0, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }
  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ymax = params[f'ymax_ds{ds_index}'].value
    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    Abulk = c0

    #ODE system
    if include_mtl:
      [LA, LAA, Asurf] = y
      L = ymax - LA - LAA
      
      dAsurf = kt*(Abulk - Asurf) - (ka1*L*Asurf - kd1*LA) - (ka2*LA*Asurf - kd2*LAA)   
      dLA = ka1*L*Asurf - kd1*LA - (ka2*Asurf*LA - kd2*LAA)
      dLAA = ka2*LA*Asurf - kd2*LAA
      
      return [dLA, dLAA, dAsurf]
    
    else:
      [LA, LAA] = y
      L = ymax - LA - LAA
      
      dLA = ka1*L*Abulk - kd1*LA - (ka2*Abulk*LA - kd2*LAA)
      dLAA = ka2*LA*Abulk - kd2*LAA
      
      return [dLA, dLAA]

class Triple_sites:
#no mtl for now
#note: that this model assumes that the binding of the analyte to one subunit of the ligand 
# does not affect the binding of the analyte to other subunits of the ligand, and also 
# assumes that the binding site on each subunit is independent.

  name = 'Triple_sites'

  no_ODEs = 3
  signal_components = 3

  def __init__(self):
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e+04, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'ka3' : Parameter(name = 'ka3', vary = True, value = 1e+04, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd3' : Parameter(name = 'kd3', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': 's-1'}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax1' : Parameter(name = 'ymax1', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'ymax2' : Parameter(name = 'ymax2', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'ymax3' : Parameter(name = 'ymax3', vary = True, value = 1, min = 1e-12, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = -1, min = 0, max = 1, user_data = {'type': 'local_step', 'units': 'RU'}),
    }

  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ka3 = params['ka3'].value
    kd3 = params['kd3'].value
    try:
      ymax1 = params[f'ymax1_ds{ds_index}'].value
      ymax2 = params[f'ymax2_ds{ds_index}'].value
      ymax3 = params[f'ymax3_ds{ds_index}'].value
    except:
      ymax1 = params[f'ymax1'].value
      ymax2 = params[f'ymax2'].value
      ymax3 = params[f'ymax3'].value

    if 'kt' in params: 
      include_mtl = True
      kt = params['kt'].value
    else: 
      include_mtl = False
    Abulk = c0



    [L1A, L2A, L3A] = y
    L1 = ymax1 - L1A
    L2 = ymax2 - L2A
    L3 = ymax3 - L3A
    
    dL1A = ka1*L1*Abulk - kd1*L1A
    dL2A = ka2*L2*Abulk - kd2*L2A
    dL3A = ka3*L3*Abulk - kd3*L3A
    
    return [dL1A, dL2A, dL3A]


class Trivalent_ligand:
  name = 'trivalent_ligand'

  no_ODEs = 3
  signal_components = 3

  def __init__(self) -> None:
    self.params_definitions = {
      'ka1' : Parameter(name = 'ka1', vary = True, value = 1e+04, min = 1e+01, max = 1e+07, user_data = {'type': 'global', 'units': 'M-1s-1'}),
      'kd1' : Parameter(name = 'kd1', vary = True, value = 1e-02, min = 1e-06, max = 1e-01, user_data = {'type': 'global', 'units': 's-1'}),
      'ka2' : Parameter(name = 'ka2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': ''}),
      'kd2' : Parameter(name = 'kd2', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': ''}),
      'ka3' : Parameter(name = 'ka3', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': ''}),
      'kd3' : Parameter(name = 'kd3', vary = True, value = 1e-02, min = 1e-07, max = 1e+07, user_data = {'type': 'global', 'units': ''}),
      'kt' : Parameter(name = 'kt', vary = True, value = 1e+07, min = 1e+01, max = 1e+12, user_data = {'type': 'global', 'units': 'RU M-1s-1'}),
      'ymax' : Parameter(name = 'ymax', vary = True, value = 1, min = 0, max = 1000, user_data = {'type': 'local_dataset', 'units': 'RU'}),
      'offset' : Parameter(name = 'offset', vary = True, value = 0, min = -1000, max = 1000, user_data = {'type': 'local_step', 'units': 'RU'}),
    }
  def __repr__(self):
    return self.name

  def ydot(self, t, y, params, c0, ds_index):
    ka1 = params['ka1'].value
    kd1 = params['kd1'].value
    ka2 = params['ka2'].value
    kd2 = params['kd2'].value
    ka3 = params['ka3'].value
    kd3 = params['kd3'].value

    try:
        ymax = params[f'ymax_ds{ds_index}'].value
    except:
        ymax = params['ymax'].value
    
    [LA, LAA, LAAA] = y
    L = ymax - LA - LAA - LAAA
    Abulk = c0

    dLA = ka1*L*Abulk - kd1*LA - ka2*LA*Abulk + kd2*LAA 
    dLAA = ka2*LA*Abulk - kd2*LAA -ka3*LAA*Abulk + kd3*LAAA
    dLAAA = ka3*LAA*Abulk - kd3*LAAA

    
    return [dLA, dLAA, dLAAA]

def create_params(exp, model, mtl = False, offsets = False, ymax = 'local_dataset'):
  """Returns params for a given model \
      must receive info on exp structure including number of datasets and steps."""
          
  params = Parameters()

  params_definitions = model.params_definitions
   
  for parname, par in params_definitions.items():
    if par.name == 'kt' and mtl == False: continue
    if par.name == 'offset' and offsets == False: continue
    if 'ymax' in parname and ymax != 'local_dataset':
      if ymax not in ['global', 'local_dataset']:
        raise ValueError('ymax can only be fitted globaly or localy')
      par.user_data['type'] = ymax


    if par.user_data['type'] == "global":
      params.add(par)

    if par.user_data['type'] == "local_dataset":
      for dataset in exp.datasets:
        ds_par = copy(par)
        i = dataset.index
        ds_par.name = f"{ds_par.name}_ds{i}"
        params.add(ds_par)
    
    if par.user_data['type'] == "local_step":
      for dataset in exp.datasets:
        i = dataset.index
        for step in dataset.steps:
          j = step.index
          step_par = copy(par)
          # if step.index == 0:
          #     offset_val = 0 - dataset.response[step.start]
          #     step_par.set(value=-offset_val)
          # else: 
          #     offset_val = dataset.response[dataset.steps[step.index - 1].stop] - dataset.response[step.start]
          #     step_par.set(value=-offset_val)
          step_par.name = f"{step_par.name}_ds{i}_step{j}"
          params.add(step_par)

  return params

  # def summary(self, exp=None):
  #   if exp is not None and exp.params is not None:
  #     params = exp.params
  #     s = [] 
  #     s.append(f"ka (1/Ms): {params['ka'].value:.3E}\n")
  #     s.append(f"kd (1/s): {params['kd'].value:.3E}\n")
  #     if 'kt' in params:
  #       s.append(f"kt (RU M-1 s-1): {params['kt'].value:.3E}\n")
  #     s.append(f"KD (M):{params['kd'].value/params['ka'].value:.3E}\n")
  #     y = np.var(np.concatenate([dataset.response for dataset in exp.datasets]).flatten(), ddof=2)
  #     s.append(f"R²: {1 - exp.result.redchi / y:.5f}\n")
  #     s.append(f"chi²: {exp.result.chisqr}\n")
  #     s.append("See Report page for additional info")
  #     return s

