import models
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from lmfit import Minimizer, conf_interval, Parameters
import matplotlib.pyplot as plt
import pickle
import corner
import numdifftools
from copy import deepcopy
import json
import base64

class Step:
  def __init__(self, index: int, dataset_index: int, start: float, stop: float, concentration: float = -1):
    self.start = start
    self.stop = stop
    self.concentration = concentration
    self.dataset_index = dataset_index
    self.index = index
    self.type = None
  
  @property
  def duration(self):
    return self.stop - self.start
  
  def __repr__(self):
    return f"{type(self).__qualname__}({self.index}, start = {self.start}, stop = {self.stop}, len = {self.duration}, c = {self.concentration}, type = {self.type})"

  def __str__(self):
    return self.__repr__()

class Dataset:
  def __init__(self, index: int, t: np.array, response: np.array):
    self.index = index
    self.name = ''
    self.t = t
    self.response = response
    self.use_for_fit = True
    self.fit_response = np.zeros_like(response)
    self.steps: list = []
    self.baseline_start = 0
    self.baseline_stop = 0

  @property
  def no_steps(self):
    return len(self.steps)

  def __repr__(self):
    extras = ''
    if "baseline_start" in self.__dict__ and "baseline_stop" in self.__dict__:
      extras += f", baseline = ({self.baseline_start}, {self.baseline_stop})"
    if "use_for_fitting" in self.__dict__:
      extras += f", use_for_fit = {self.use_for_fit}"
    return f"{type(self).__qualname__}({self.index}, name = {self.name}, no_steps = {self.no_steps}, len = {self.t[-1] - self.t[0]}{extras})"

  def __str__(self):
    return self.__repr__()

  def __iter__(self):
    yield from self.steps

  def add_step(self, start: float, stop: float, concentration: float = -1):
    self.steps.append(Step(self.no_steps, self.index, start, stop, concentration))
  
  def add_many(self, steps):
    for step in steps:
      start, stop, concentration = step
      self.add_step(start, stop, concentration)
 
class Exp:
  def __init__(self):
    self.datasets: list = []
    self.model = None
    self.params = None
    # self.jumps: list = [] #only for simulation
    # self.offsets: list = []
    self.result = None
    self.info = '' #optional

  @property
  def no_datasets(self):
    return len(self.datasets)

  def __repr__(self):
    if self.model is not None:
      name = self.model.name
    else: 
      name = 'NA'
    if self.params is not None:
      params = len(self.params)
    else:
      params = 'NA'
    header = f"<Exp>: Model = <{name}> ({params} params), Datasets = {self.no_datasets}, fitted = {'No' if self.result is None else 'Yes'}\n"

    s = []
    s.append(header)
    for ds in self.datasets:
      s.append(f"\t<{ds.__repr__()}>\n")
      for step in ds.steps:
        s.append(f"\t\t<{step.__repr__()}>\n")
    
    return ''.join(s)
  
  def __iter__(self):
    yield from self.datasets

  def add_dataset(self, t: np.array, response: np.array):
      self.datasets.append(Dataset(self.no_datasets, t, response))
  
  def get_models(self):
    return models.get_models_dict().keys()

  def set_model(self, model: str):
    """
    Sets model for experiment object.
    
    Must be one of:
    - 'One to one', 
    - 'Bivalent analyte', 
    - 'Conformational change', 
    - 'Heterogeneous ligand', 
    """
    self.model = models.get_models_dict()[model]()

  def create_params(self, **kwargs):
    """
    Creates parameters for a given binding model. 
    
    Parameters
    ----------
    offsets : bool, optional
        Whether to allow offsets for each dataset (default is False).
    mtl: bool, optional
        Whether to include mass transport limitation (default is False).
    """

    if self.model is None:
      raise ValueError("Experiment model is 'None'. Specify it with Exp.set_model('model') first.")
    else:
      self.params = models.create_params(self, self.model, **kwargs)

  def datasets_from_dataframe(self, df, **kwargs):
    for col in df: 
      self.add_dataset(name = col)
    return self
  
  def get_ds_index_by_name(self, name):
    for i, dataset in enumerate(self.datasets):
      if dataset.name == name:
        return i

  def interstep_correction(self):
    for dataset in self.datasets:
      offset0 = dataset.response[np.isfinite(dataset.response) & (dataset.t >= dataset.steps[0].start)][0]
      dataset.response[(dataset.t >= dataset.steps[0].start)] = dataset.response[(dataset.t >= dataset.steps[0].start)] - offset0
      
      for step in dataset.steps[1:]:
        left = dataset.response[np.isfinite(dataset.response) & (dataset.t < step.start)]
        if left.size == 0:
            left = 0
        else:
            left = left[-1]
        right = dataset.response[np.isfinite(dataset.response) & (dataset.t >= step.start)]
        if right.size == 0:
            right = 0
        else:
            right = right[0]
        offset = right - left
        dataset.response[(dataset.t >= step.start)] = dataset.response[(dataset.t >= step.start)] - offset

  def crop(self, time, show=False, **kwargs):
    if show==True:
      exp_copy = deepcopy(self)

      for dataset in exp_copy.datasets:
        for step in dataset:
          start = step.start
          stop = step.start + time
          dataset.response[(dataset.t>=start) & (dataset.t<stop)] = np.nan
      
      return exp_copy.plot(**kwargs)
    
    else:
      for dataset in self.datasets:
        for step in dataset:
          start = step.start
          stop = step.start + time
          dataset.response[(dataset.t>=start) & (dataset.t<stop)] = np.nan


  def calc_dataset_response_from_params(self, dataset, params=None, y0=None):
    if params is None: params = self.params
    if y0 is None: #create y0 if not explicitly passed
      if 'kt' not in params : 
        y0=[0 for ODE in range(self.model.no_ODEs)]
      else:  
        y0=[0 for ODE in range(self.model.no_ODEs + 1)] #if MTL is included it increases number of ODEs to solve
    
    
    calculated_dataset_response = np.zeros_like(dataset.t)
    for i, step in enumerate(dataset.steps):
      if i < dataset.no_steps - 1:
        mask = (dataset.t>=step.start) & (dataset.t<dataset.steps[i+1].start)
        t_step = dataset.t[mask]
        next_t0 = dataset.steps[i+1].start
        t_step = np.concatenate((t_step, [next_t0]))
      else:
        mask = (dataset.t>=step.start)
        t_step = dataset.t[mask]

      c0 = step.concentration
      if f"offset_ds{dataset.index}_step{step.index}" in params:
        offset = params[f"offset_ds{dataset.index}_step{step.index}"].value
      else:
        offset = 0
      
      responses = odeint(self.model.ydot, y0, t_step, args=(params, c0, dataset.index), tfirst=True, rtol=1e-12, atol=1e-12) 
      
      if i < dataset.no_steps - 1:
        y0 = responses[-1,:].flatten() #new y0 for next step
        responses[:,0] = responses[:,0] + offset
        responses = np.sum(responses[:,0:self.model.signal_components], axis=1)
        calculated_dataset_response[mask] = responses[:-1]
      else:
        responses[:,0] = responses[:,0] + offset
        responses = np.sum(responses[:,0:self.model.signal_components], axis=1)
        calculated_dataset_response[mask] = responses
    
    return calculated_dataset_response

  def recalculate_fit_response(self, params=None):
    for dataset in self.datasets:
      dataset.fit_response = self.calc_dataset_response_from_params(dataset, params=None)

  def simulate_response(self, params=None, noise=0., offset=0.):

    if params is None: params = self.params
    for dataset in self.datasets:
      response = self.calc_dataset_response_from_params(dataset, params)
      t = dataset.t
      dataset.response = response + np.random.normal(0, noise, len(response))
      if offset != 0: #remove when done testing
        for step in dataset.steps:
          jump = np.random.normal(0)*offset
          self.jumps.append(jump) #remove when done testing
          mask = (t>=step.start)&(t<step.stop)
          dataset.response[mask] = dataset.response[mask] + jump

  def residuals(self, params, y0=None):
    resids = []
    for dataset in self.datasets:
      if dataset.use_for_fit == False: 
        continue
      dataset_resids = dataset.response - self.calc_dataset_response_from_params(dataset, params, y0)
      resids.append(dataset_resids)
    return np.concatenate(resids)

  def prep_for_fit(self):
    for dataset in self.datasets:
      if all(step.concentration<=0 for step in dataset.steps): 
        print(f'Dataset {dataset.index} will be ommited. It does not contain any step with concentration>0.')
        dataset.use_for_fit=False
        #fix offsets if dataset is not used 
        for step in dataset.steps:
          offset_name = f"offset_ds{dataset.index}_step{step.index}"
          if offset_name in self.params:
            self.params[offset_name].set(vary=False)
      else:
        for step in dataset.steps:
          offset_name = f"offset_ds{dataset.index}_step{step.index}"
          if offset_name in self.params:
            self.params[offset_name].set(vary=True)

      if all(np.isfinite(dataset.response))!=True:
        print(f'Nan and inf are not valid responses. Some values in dataset {dataset.index} are not finite. LMFIT is set to omit.')

        
  def fit_params(self, y0=None, method='least_squares', **kwargs):
    if self.model is None:
      print("No model was set. Using 'One to one'")
      self.model = models.One_to_one()
    
    if self.params is None:
      print("No params were set. Using defaults (no mtl, no offsets")
      self.params = models.create_params(self, self.model)
    
    self.prep_for_fit()
    
    print(f"fitting using: {method}")
    mini = Minimizer(self.residuals, self.params, nan_policy='omit')
    result = mini.minimize(method=method)
    ## result = minimize(myfunc, ....., maxfev=100) https://groups.google.com/g/lmfit-py/c/M_t2W3Z6H50
    self.result = result
    self.params = result.params
    self.recalculate_fit_response()
    return result.message #remove when done testing


  def format_si_prefix(self, n):
    """
    Format a number using an SI prefix and return a string representation of the number with an SI prefix
    """
    prefixes=['a', 'f', 'p', 'n', 'µ', 'm', '', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    unit='M'

    sign = np.sign(n)
    absn = abs(n)
    i = 6 # n >=1 and n<1000

    if absn == 0:
        return '0'
    elif absn>0 and absn<=1:
        while absn<1:
            absn *= 1000
            i -= 1
    elif absn>=1 and absn<1000:
        return f"{sign*absn}{unit}"
    else:
        while absn >= 1000:
            absn /= 1000
            i += 1
    
    if i>=0 and i<len(prefixes):
        return f"{sign*absn:.3f}{prefixes[i]}{unit}"
    else:
        print('#', sign, absn, i)
        return f"{n:.3e}{unit}"


  def calculate_confidence_intervals(self): 
      mini = Minimizer(self.residuals, self.params, nan_policy='omit')
      result = self.result
      ci = conf_interval(mini, result, verbose=True)
      return ci

  def posterior_probability(self, burn=300, steps=500, thin=20, is_weighted=False, progress=True):
      """
      Returns posterior probability distribution of parameters calculated with mcee
      """
      params = self.params.copy()

      mini = Minimizer(self.residuals, params, nan_policy='omit')
      result = mini.minimize(method='emcee', burn=burn, steps=steps, thin=thin, is_weighted=is_weighted, progress=progress)
    
      return result

  def plot_corner(emcee_result):
      """
      Plots the parameter covariances returned by emcee using corner.
      Returns corner figure.
      """

      fig = corner.corner(emcee_result.flatchain, 
                          labels=emcee_result.var_names,
                          truths=list(emcee_result.params.valuesdict().values()),
                          quantiles=(0.16, 0.84),
                          use_math_text=True,)
      
      #fig.savefig('./output/emcee_plot.png')
      
      return fig

  def format_label(self, dataset, labels):
    """Return a string consisting of dataset name with or without step concentrations"""
    s = []
    s.append(dataset.name)
    for step in dataset:
      if step.concentration > 0:
        s.append(self.format_si_prefix(step.concentration))
    
    if labels == 'names':
      return s[0]
    elif labels == 'concentrations':
      return ' '.join(s[1:])
    else:
      if labels != 'both': print(f'Unknown argument {labels}. Reverting to both.')
      return ' '.join(s)
      

  def plot(self, fit = False, correct_offsets = False, use = 'pyplot', labels = 'both'):
    """
        Return sensorgram figure.
    
        Parameters
        ----------
        fit : boolean
            whether fitted curves should be included
        correct_offsets : boolean
            whether offsets should be corrected
        use : string
            what backend should be used for plotting
            must be one of: 'pyplot', 'plotly'
            default is 'pyplot'
        labels : string
            determines labels on the plot.
            must be one of: 'names' (dataset names), 
                            'concentrations' (list of non zero concentrations used in dataset)
                            'both'

        Returns
        -------
        Figure object.
    """

    if self.params is None:
      print("Exp.params is None. Fit and offset plots are not available")
      fit = False
      correct_offsets = False
    else: params = self.params

    if use == 'pyplot':
      fig, ax = plt.subplots()
      for dataset in self.datasets:
        if dataset.use_for_fit == False: continue
        range_mask = (dataset.t>=dataset.steps[0].start) & (dataset.t<=dataset.steps[-1].stop)
        label = self.format_label(dataset, labels)
        if correct_offsets:
          response = np.copy(dataset.response)
          for step in dataset.steps:
            mask = (dataset.t>=step.start) & (dataset.t<step.stop)
            offset = params[f'offset_ds{dataset.index}_step{step.index}'].value
            response[mask] = response[mask] - offset
        else: response = dataset.response
        
        ax.plot(dataset.t[range_mask], response[range_mask], label=label)
      
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      
      if fit==False: 
        return (fig, ax) 

      #plot fitted response
      self.recalculate_fit_response()
      for dataset in self.datasets:
        if dataset.use_for_fit == False: continue
        range_mask = (dataset.t>=dataset.steps[0].start) & (dataset.t<=dataset.steps[-1].stop)
        label = f"Fitted {dataset.index}"
        if correct_offsets:
          fit_response = np.copy(dataset.fit_response)
          for step in dataset.steps:
            mask = (dataset.t>=step.start) & (dataset.t<step.stop)
            offset = self.params[f'offset_ds{dataset.index}_step{step.index}'].value
            fit_response[mask] = fit_response[mask] - offset
        else: fit_response = dataset.fit_response
        
        ax.plot(dataset.t[range_mask], fit_response[range_mask], color='black', label= label)
        title = [f"{par.split('_')[0]}: {val.value:.1E}" for par, val in self.params.items() if ('k' in par) or ('ymax' in par)]
        title = sorted(set(title))
        title = ' '.join(title)
        plt.title(title, fontsize=12)

      return (fig, ax)

    if use == 'plotly': pass

  def plot_resids(self):
    fig, axs = plt.subplots(self.no_datasets + 1)
    fig.suptitle('Residual plots', fontsize=24)
    ax_limit = max([dataset.t[-1] for dataset in self.datasets])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #all datasets
    for dataset in self.datasets:
      resid = dataset.response - dataset.fit_response
      axs[0].plot(dataset.t, resid)
      axs[0].set_title(f"All datasets")
    axs[0].axhline(y = 0, color = 'black', linestyle = '-')
    axs[0].set_xlim([0, ax_limit])

    #seperate subplots
    for i, dataset in enumerate(self.datasets):
      if i<len(colors): color = colors[i]
      resid = dataset.response - dataset.fit_response
      axs[i+1].plot(dataset.t, resid, color=color)
      axs[i+1].axhline(y = 0, color = 'black', linestyle = '-')
      axs[i+1].set_xlim([0, ax_limit])
      concs = ' '.join([f"{c:.1E}" for c in dataset.concentrations])
      axs[i+1].set_title(f"D{i}: {concs}")
    
    for ax in axs.flat:
      ax.label_outer()

    fig.subplots_adjust(hspace=0.5)

  def to_dataframe(self, include_fit=False, correct_offsets=False):
    df = pd.DataFrame([])
    for dataset in self.datasets:
      if correct_offsets:
        response = np.copy(dataset.response)
        fit_response = np.copy(dataset.fit_response)
        for step in dataset.steps:
          mask = (dataset.t>=step.start) & (dataset.t<step.stop)
          offset = self.params[f'offset_s{step.index}_{dataset.index}'].value
          response[mask] = response[mask] - offset
          fit_response[mask] = fit_response[mask] - offset
      else:
        response = dataset.response
        fit_response = dataset.fit_response


      label = f"{dataset.index}: " + ' '.join([f"{step.concentration:.1E}" for step in dataset.steps])
      df[label] = pd.Series(data=response, index=dataset.t)
      
      if include_fit:
        label_fitted = f"fitted {dataset.index}"
        df[label_fitted] = pd.Series(data=fit_response, index=dataset.t)
    
    return df

  def to_pickle(self, fname=None):
    if fname == None:
      fname = f"exp_{self.model.name.replace(' ', '_')}.pkl"
    
    with open(fname, 'wb') as handle:
      pickle.dump(self, handle)

  def to_dict(self):
    """Return dict representation of Experiment."""

    d = {}
    
    try:
      d['model'] = self.model.name
    except:
      d['model'] = None
    
    try:
      d['params'] = self.params.dumps() #params have their own method
    except:
      d['params'] = None

    
    d['info'] = self.info
    
    d['datasets'] = []
    for dataset in self.datasets:
      ds = {}
      ds['index'] = dataset.index
      ds['name'] = dataset.name if dataset.name != '' else str(dataset.index)
      ds['t'] = base64.b64encode(dataset.t).decode('utf-8')
      ds['response'] = base64.b64encode(dataset.response).decode('utf-8')
      ds['use_for_fit'] = dataset.use_for_fit
      ds['baseline_start'] = dataset.baseline_start
      ds['baseline_stop'] = dataset.baseline_stop
      ds['steps'] = []
      
      for step in dataset.steps:
        st = {}
        st['start'] = step.start
        st['stop'] = step.stop
        st['concentration'] = step.concentration
        st['dataset_index'] = step.dataset_index
        st['index'] = step.index
        st['type'] = step.type
        ds['steps'].append(st)
      
      d['datasets'].append(ds)
    
    return d

  def dumps(self, **kwargs):
    """Represent Experiment as a JSON string.
        Parameters
        ----------
        **kwargs : optional
            Keyword arguments that are passed to `json.dumps`.
        Returns
        -------
        str
            JSON string representation of Parameters.
    """
    d = self.to_dict()
    return json.dumps(d, **kwargs)


  def dump(self, f, **kwargs):
    """Write JSON representation of Experiment to a file-like object.
        Parameters
        ----------
        f : file-like object
            An open and `.write()`-supporting file-like object.
        **kwargs : optional
            Keyword arguments that are passed to `dumps`.
        Returns
        -------
        int
            Return value from `f.write()`: the number of characters
            written.
    """
    return f.write(self.dumps(**kwargs))

  def save(self, fname, **kwargs):
    with open(fname, 'w') as handle:
      self.dump(handle)


  def loads(self, s, **kwargs):
    """Load Experiment from a JSON string
    
        Parameters
        ----------
        s : json string
        **kwargs : optional
            Keyword arguments that are passed to `json.loads`.
        Returns
        -------
        Experiment object.
    """
    d = json.loads(s, **kwargs)
    
    model_name = d['model']
    if model_name is not None:
      self.model = models.get_models_dict()[model_name]()
    
    params_string = d['params']
    if params_string is not None:
      self.params = Parameters()
      self.params.loads(params_string)

    self.info = d['info']

    for dataset in d['datasets']:
      index = dataset['index']
      if index != self.no_datasets:
        print('dataset index does not match!')
      name = dataset['name']
      t = np.frombuffer(base64.b64decode(dataset['t']))
      response = np.frombuffer(base64.b64decode(dataset['response']))
      baseline_start = dataset['baseline_start']
      baseline_stop = dataset['baseline_stop']
      use_for_fit = dataset['use_for_fit']

      self.add_dataset(t, response)
      self.datasets[-1].baseline_start = baseline_start
      self.datasets[-1].baseline_stop = baseline_stop
      self.datasets[-1].name = name
      self.datasets[-1].use_for_fit = use_for_fit

      for step in dataset['steps']:
        start = step['start']
        stop = step['stop']
        concentration = step['concentration']
        ds_index = step['dataset_index']
        index = step['index']
        if index != self.datasets[-1].no_steps:
          print('step index does not match!')
        stype = step['type']

        self.datasets[-1].add_step(start, stop, concentration)
        self.datasets[-1].steps[-1].type = stype


  def load(self, f, **kwargs):
    """Load JSON representation of Experiment from a file-like object.
        
        Parameters
        ----------
        f : file-like object
            An open and `.read()`-supporting file-like object.
        **kwargs : optional
            Keyword arguments that are passed to `loads`.
        
        Returns
        -------
        Experiment object
    """
    self.loads(f.read(), **kwargs)

  def load_bli_data(self, files):
        df_list = []
        for i, f in enumerate(files):
            try:
              name = f.name
            except:
              name = f.split('\\')[-1]
            df = pd.read_csv(f, sep=',', skipinitialspace=True, index_col=0, names=['t', name, f'{i}_steps'], skiprows=1)
            df_list.append(df)

        if len(df_list)>1: df = pd.concat(df_list, axis=1)
        else: df = df_list[0]

        #prep exp
        exp = self
        for i, col_name in enumerate(df):
            if i%2 == 1: continue
            ds_index = int(i/2)

            exp.add_dataset(t = df.index.to_numpy(), 
                            response = df[col_name].to_numpy())
            exp.datasets[-1].name = col_name
            exp.datasets[-1].use_for_fitting = True
            
            step_names = df.iloc[:,i+1].unique()
            step_names = step_names[~pd.isnull(step_names)]

            #assumes last three steps are bs, as and ds
            for j in range(len(step_names)-3, len(step_names)):
                #blitz csv have time(index), response(col1) and steps(col2) columns
                loc_start = df.iloc[:,i+1].eq(step_names[j]).argmax() 
                if j < len(step_names) - 1 :
                    loc_stop = df.iloc[:,i+1].eq(step_names[j+1]).argmax()
                    stop = df.index[loc_stop]

                else:
                    loc_stop = df.index.size-1 
                    stop = round(df.index[loc_stop])
                
                start = df.index[loc_start]
            
                if j == len(step_names)-3:
                    exp.datasets[ds_index].baseline_start = start
                    exp.datasets[ds_index].baseline_stop = stop
                else:
                    exp.datasets[ds_index].add_step(start=start, stop=stop, concentration=0)
                    if j == len(step_names)-2: 
                      exp.datasets[ds_index].steps[-1].type = 'Association'
                    if j == len(step_names)-1: exp.datasets[ds_index].steps[-1].type = 'Dissociation'

        #df = df[[col for col in df if 'ds' in col]] #discard 'steps' column



if __name__ == '__main__': 
  pass