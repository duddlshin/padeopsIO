import os
import warnings

# try: 
#     import f90nml
# except ImportError: 
#     warnings.warn("Could not find package f90nml in system path. ")
#     # TODO - handle this somehow
#     f90nml = None

from padeopsIO.turbine import Turbine
from padeopsIO.nml_utils import parser
    
class TurbineArray(): 
    """
    # TODO fix class docstring
    """
    
    def __init__(self, turb_dir=None, num_turbines=None, 
                 init_ls=[], init_dict=None, 
                 ADM_type=None, verbose=False, sort='xloc'): 
        """
        Constructor function for a TurbineArray class
        
        Parameters
        ----------
        turb_dir (path-like) : path to a turbine array directory in PadeOps
        num_turbines (int) : optional, number of turbines. Default: number 
            of turbine files in turbine directory
        ADM_type (int) : ADM type. Default: 2. 
        verbose (bool) : additional print statements
        
        Returns
        -------
        TurbineArray class instance
        """

        # this initializes from a dictionary output by todict()
        self.verbose = verbose
        self._sort_by = sort
        if init_dict is not None: 
            self.fromdict(init_dict)

            if self.verbose: 
                print("TurbineArray: Initialized from", turb_dir)
            return
        
        self.turb_dir = turb_dir
                
        if turb_dir is not None: 
            # glean namelist inputs from the turbine directory

            if len(init_ls) == 0: 
                # begin reading in turbines
                filenames = os.listdir(turb_dir)
                filenames.sort()  # sort these into ascending order
                if self.verbose: 
                    print("Reading turbines from the following files:\n", filenames)

                for i, filename in enumerate(filenames): 
                    turb_nml = parser(os.path.join(turb_dir, filename))
                    init_ls.append(turb_nml)
            elif self.verbose: 
                print('__init__(): `turb_dir` superceded by `init_ls` kwarg.')

        if num_turbines is not None: 
            self.num_turbines = num_turbines
            
            if num_turbines != len(init_ls):  
                warnings.warn("Not all turbines in the specified directory will be used")

                if self.verbose: 
                    print("\tRequested {:d} turbines, but found {:d} files.".format(num_turbines, len(init_ls)))
        else: 
            self.num_turbines = len(init_ls)
            
        # for now, each turbine can simply be a dictionary appended to a list
        self.array = []  # array is deprecated (06/01/2023)
        self.turbines = []

        for i, turb_nml in enumerate(init_ls): 
            if i >= self.num_turbines: 
                break  # only read in up to num_turbines turbines
                
            self.array.append(turb_nml)
            self.turbines.append(Turbine(turb_nml, verbose=self.verbose, n=i+1, sort=self._sort_by))
            
            if self.verbose: 
                print("\tTurbineArray: added turbine to array form file", filename)
                
        # sort the turbine array: 
        self.sort()
        
        if self.num_turbines == 1: 
            # make the variables more accessible
            if self.verbose: 
                print("\tAdding convenience variables...")

            for key in self.turbines[0].input_nml['actuator_disk'].keys(): 
                self.__dict__[key] = self.turbines[0].input_nml['actuator_disk'][key]
        
        if self.verbose: 
            print("TurbineArray: Initialized from", turb_dir)
            
            
    def sort(self, reverse=False): 
        """
        Sorts the `turbines` property according to self._sort_by. 
        
        NOTE: This does not rearrange the `array` property, which is deprecated. 
        """
        self.turbines.sort(reverse=reverse)
        
        
    def set_sort(self, sort_by, sort=True, reverse=False): 
        """
        Sets the sorting field for all turbines in the array. 
        
        Sorts the array if sort=True (default True)
        """
        
        for turbine in self.turbines: 
            turbine.set_sort(sort_by)
            
        self._sort_by = sort_by
        if sort: 
            self.sort(reverse)
        
        
    def __iter__(self): 
        """
        Iterates through the turbine list. 
        """
        return self.turbines.__iter__()
        
    
    def fromdict(self, init_dict): 
        """
        Converts a dictionary object given by todict() back into a TurbineArray object. 
        """
        for key in init_dict.keys(): 
            self.__dict__[key] = init_dict[key]

    
    def todict(self): 
        """
        Converts self.__dict__ into a dictionary with no namelists. 
        """
        ret = self.__dict__.copy()
        if 'turbines' in ret.keys(): 
            ret['turbines'] = [t.input_nml for t in ret['turbines']]  # save input namelists
        return ret
    
    
    def __str__(self): 
        """
        Overrides the default object print statement. 
        """
        return "Turbine array object at {:s} with {:d} turbines".format(self.turb_dir, self.num_turbines)
        
        

if __name__ == "__main__": 
    """
    TODO - add unit tests to class
    """
    print("turbineArray.py: No unit tests included yet. ")
    
