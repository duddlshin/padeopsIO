"""
Export and data copy functions for PadeopsIO

Command line usage: 
>>> python export.py <directory>

Kirby Heck
2024 July 15
"""

import shutil  # we will use this to copy files
import time
import glob
import os
import sys
from pathlib import Path

from .. import BudgetIO


def copy_padeops_data(
        case=None, 
        case_dir=None, 
        export_dir=None,
        runid=1,
        tidx=None,
        copy_budgets=True,
        copy_restarts=True,
        copy_fields=True,
        copy_logfiles=True,
        overwrite=False,
        quiet=False
    ): 
    """
    Copy a subset of PadeOps data to a new location
    
    Parameters
    ----------
    case : BudgetIO object, optional
        if `case` is not given, then `case_dir` must be provided. 
    case_dir : path-like
        Path to PadeOps source data
    export_dir : path-like
        Destination to copy files. Defaults to `case_dir.parent / 'export' / case.filename`
    runid : int, optional
        if `case_dir` is provided, looks for runid. Defaults to 1. 
    tidx : int, optional
        Uses case.unique_tidx()[-1] if None. Default None
    copy_budgets : bool, optional
        Copies budget files if True. Default True. 
    copy_restarts : bool, optional
        Copies restart files if True; restart files must be locatable. Default True. 
    copy_fields : bool, optional
        Copies final field file dump, if true. Default True. 
    copy_logfiles : bool, optional
        copies logfiles (ending in *.[oe][0-9]*). Default True. 
    overwrite : bool, optional
        If True, rewrites existing files. Default False. 
    fname : str, optional
        Formatted string. Files will be copied into a new directory named
        fname.format(case.filename). Default: '{:s}'
    quiet : bool, optional
        Silences print statements. Default false. 
    """
    time_st = time.perf_counter()  # start timer

    # find source data
    if case is None: 
        case = BudgetIO(case_dir, padeops=True, runid=runid, quiet=quiet)
        case_dir = Path(case_dir)
    else: 
        case_dir = Path(case_dir)
    
    # set export directory
    if export_dir is None: 
        target = case_dir.parent / 'export' / case.filename

    if not quiet: 
        print('Copying files. Target directory: ', target)

    # make the target directory, if needed
    try: 
        os.makedirs(target)
    except FileExistsError as e: 
        pass  # directory already exists

    # glean last tidx and last budget_tidx if no `tidx` is explicitly given
    if tidx is None: 
        last_tidx = case.unique_tidx(return_last=True)
        if copy_budgets: 
            last_budget_tidx = case.unique_budget_tidx(return_last=True)
    else: 
        last_tidx = tidx
        last_budget_tidx = tidx  # assume these exist

    if not quiet:
        print(f'Selected the following TID to copy: {last_budget_tidx:.0f} for budgets, {last_tidx:.0f} for fields')

    # find all of the files... hopefully they are saved in the right place! 
    files = {}
    if copy_budgets: 
        files['budgets'] = list(case_dir.glob(f'Run{case.runid:02d}*budget*_t{last_budget_tidx:06d}*'))

    if copy_fields: 
        files['fields'] = list(case_dir.glob(f'Run{case.runid:02d}*_t{last_tidx:06d}.out'))

    if copy_restarts: 
        r_tidx = case.input_nml['input']['restartfile_tid']
        r_id = case.input_nml['input']['restartfile_rid']
        r_dir = Path(case.input_nml['input']['inputdir'])
        files['restarts'] = list(r_dir.glob(f'RESTART_Run{r_id:02d}*.{r_tidx:06d}'))
    if copy_logfiles: 
        files['logfiles'] = list(case_dir.glob('*.[oe][0-9]*'))

    # IO
    files['input'] = list(case_dir.glob('*.dat'))

    if case.associate_turbines: 
        files['turbine'] = [Path(case.input_nml['windturbines']['turbinfodir'])]
        files['power'] = list(case_dir.glob('*.pow'))
        files['disk_vel'] = list(case_dir.glob('*.vel'))

    all_files = sum(files.values(), [])  # concatenate all the lists into one list

    # copy the files
    if not quiet:
        print('Total number of files to copy: ', len(all_files))

    existing_files = [f.name for f in Path(target).glob('*')]  # [os.path.basename(file) for file in os.listdir(target)]  # existing files
    n_skip = 0 

    # copy files! 
    for f in all_files: 
        # for each file, check to see if it already exists (will save some time)
        if f.name in existing_files and not overwrite: 
            n_skip += 1
            continue  # skip this one

        # we can't copy directories in the same way that we copy files
        if f.is_dir():  # os.path.isdir(name): 
            try: 
                shutil.copytree(f, target / f.name)
            except FileExistsError as e: 
                if not quiet:
                    print(f'\tFailed to copy `{str(f):s}`. Directory already exists.')
        else: 
            if not quiet: 
                print(f'\tCopying: {f.name:s}')  # this is the filename
            shutil.copy(f, target)

    # end the timer
    time_end = time.perf_counter()
    if not quiet: 
        print(f'Done copying for {target}. \
              Copied {len(all_files) - n_skip:d} files, \
              skipped {n_skip:d} files.')
        print(f'Elapsed time: {time_end - time_st:.1f} s\n')


def export_concurrent(dirs, export_dir, 
                      runid_primary=5, 
                      runid_precursor=4, 
                      budget_terms=None, export_kwargs=None, 
                      filetype='npz', verbose=True, 
                      copy_precursor=False, 
    ): 

    """
    Loads a list of concurrent precursor simulations from PadeOps and 
    exports them to either .npz or .mat files. 
    
    Arguments
    ---------
    dirs : list
        List of paths to load PadeOps data from
    export_dir : path-like
        Target export directory, must exist. 
    
    """
    
    # load cases: 
    cases = [BudgetIO(name, padeops=True, verbose=verbose, 
                          runid=runid_primary, 
                          normalize_origin='turb') for name in dirs]
    if copy_precursor: 
        # load precursors, same directories as cases
        pres = [BudgetIO(case.dir_name, padeops=True, verbose=verbose, 
                             runid=runid_precursor, normalize_origin=case.origin) for case in cases]

    if export_kwargs is None:  # default kwargs for exporting
        export_kwargs = {
            'overwrite': True, 
            'xlim': [-2, 25], 
            'ylim': [-2, 2], 
            'zlim': [-2, 2], 
        }  # switched to turbine-normalized coordinates
    
    export_kwargs['budget_terms'] = budget_terms
    
    if copy_precursor:
        # this is sloppy coding, please fix  #TODO 
        for case, pre in zip(cases, pres):  
            print('writing', case.filename)
            pre_name = case.filename + '_precursor'
            if filetype == 'npz': 
                case.write_npz(export_dir, **export_kwargs)
                pre.write_npz(export_dir, filename=pre_name, **export_kwargs)
            elif filetype == 'mat': 
                case.write_mat(export_dir, **export_kwargs)
                pre.write_mat(export_dir, filename=pre_name, **export_kwargs)
            else: 
                raise ValueError('export_concurrent(): `filetype` must be npz or mat. ')

    else: 
        for case in cases:  
            print('writing', case.filename)
            if filetype == 'npz': 
                case.write_npz(export_dir, **export_kwargs)
            elif filetype == 'mat': 
                case.write_mat(export_dir, **export_kwargs)
            else: 
                raise ValueError('export_concurrent(): `filetype` must be npz or mat. ')
    

def debug(): 
    case_dir = Path(r'/scratch/08445/tg877441/shear_veer/sensitivity/uniform/delta_00_ctp_00')
    copy_padeops_data(case_dir=case_dir, runid=1)

if __name__ == '__main__': 
    """
    Export from the command line. 
    """
    import sys

    debug()
        
    # # read input args: 
    # main_dir = sys.argv[1]
    
    # try: 
    #     ids = [k for k in sys.argv[2:]]
    # except: 
    #     print('export.py: Missing directory names to export')
    
    # write_dir = os.path.join(main_dir, 'export')
    
    # cases = []
    # for k in ids: 
    #     case = BudgetIO(os.path.join(main_dir, k), padeops=True, verbose=True, runid=5)
    #     cases.append(case)
        
    # copy_padeops_data(cases, write_dir) #, copy_restarts=True)
    
    # print('Done!') 
    


