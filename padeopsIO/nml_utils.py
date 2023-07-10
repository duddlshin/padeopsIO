import re

def parser(filename): 
    """
    Parses a namelist into nested dictionaries. Each key to the returned dictionary 
    is the name of a namelist in `filename`, and the value of that key is itself a 
    dictionary of namelist variables and associated values. 

    Parameters 
    ----------
    filename : path
        Path to a namelist text file

    Returns
    -------
    Namelists (nested dictionary)
    """

    Namelists = {}
    Namelist = {}  # active namelist
    Active = None
    with open(filename, 'r') as f: 
        lines = f.readlines()
        
        # loop through lines
        for line in lines: 
            res = re.search('&(\S+)', line)  # namelists start with "&"
            if res is not None:  # starting new namelist
                Active = res.group(1)  # save string in active list
                continue
            
            # within a namelist now; search `variable` = `value`
            res = re.search('(\S+)\s+=\s+(\S+)', line)
            if res is not None: 
                # found variable/value pairing
                key = res.group(1)
                value = cast_str_to_X(res.group(2))
                # TODO: Handle boolean variables too? 

                Namelist[key] = value
                continue

            res = re.search(r'/', line)
            if res is not None: 
                # Found end of Namelist
                if Active is not None: 
                    Namelists[Active] = Namelist.copy()
                    Namelist = {}
                    Active = None  # terminate namelist, start a new one

    return Namelists

def writer(filepath, Namelists): 
    """
    Writes a namelist formatted in the same way that is output by nml_parser()

    Parameters
    ----------
    filepath : path
        Path where the namelist will be written
    Namelists : Nested dict from parser()

    Returns
    -------
    None
    """
    with open(filepath, 'w') as f:  # begin writing a file
        for key in Namelists.keys(): 
            # write header
            f.write('&' + key + '\n')
            # loop thru variables
            for var in Namelists[key].keys(): 
                tmp = Namelists[key][var]
                value = cast_to_str(tmp)
                # write variable/value combinations
                f.write('{:''<24} = {:s}\n'.format(var, value))
            f.write('/\n')


def cast_str_to_X(value): 
    """
    Tries to cast the input value to an integer, then a float. 
    If all fail, then returns the original string. 

    Parameters
    ----------
    value : str
        Value read in by the parser as a string
    
    Returns
    -------
    value : str or int or float
        Cast value
    """

    # try integer casting: 
    try: 
        tmp = int(value)
        return tmp
    except ValueError: 
        pass

    # integer casting failed, try float
    try: 
        tmp1 = re.sub('[dD]', 'e', value)  # Fortran uses d, D as well as e, E for exponential notation
        tmp2 = float(tmp1)
        return tmp2
    except ValueError: 
        pass

    # TODO: add boolean 

    return value  # returns original value
    

def cast_to_str(value): 
    """
    Takes a value of unknown type (value) and casts to a fortran namelist-readable string. 

    Parameters
    ----------
    value : str or float or int
        Variable value to write into the namelist
    
    Returns
    -------
    ret : str
        String representation of the value. 
        For example: 
            int(5)      ->  '5'
            'abc'       ->  'abc'
            float(5)    ->  '5.00000000e+00'  (single precision)
            # TODO: add booleans: 
            bool(True)  ->  '.true.'
    """

    # note: this is NOT a try/except statement because we want floats (that are also integers)
    # to be parsed as floats, not integers. 
    if type(value) == int: 
        return str(value)

    try: 
        tmp = float(value)
        return '{:.08e}'.format(tmp)
    except ValueError: 
        return value  # return the original string 

