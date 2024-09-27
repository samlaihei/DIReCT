import os
from datetime import datetime

__all__ = ['construct_train_val_xlogger','read_xlogfile','xlogger','plot_lr_finder']



import os, glob, pathlib
def construct_directory(DIR_PATH,DIRNAME):
    dst_dir = os.path.join( DIR_PATH, DIRNAME)
    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)


def construct_train_val_xlogger(flname_prefix_save, experiment_prefix=None):
    # TODO: Add the github version of this lib printed somewhere in the logs 
    # TODO: in debug mode I would like to have all outputs from all workers? Or not?
    # *** update save directory with the corresponding date ***
    dirname_date = datetime.now().strftime("%d-%m-%Y::%Hh-%Mm-%Ss")
    if 'SLURM_JOBID' in os.environ.keys():
        dirname_date = "JOBID_" + os.environ['SLURM_JOBID'] +  "_" + dirname_date

    #self.params_dict[C.C_SAVE_DIRECTORY] 
    if experiment_prefix is not None:
        dirname_date = experiment_prefix+ "_on_" + dirname_date
    construct_directory(flname_prefix_save,dirname_date)
    
    # This is where EVERYTHING will be saved under 
    DIR_SAVE = os.path.join(flname_prefix_save, dirname_date)



    # XXXXXXXXXXXXXXXXxxxxx Create TRAIN/VAL loggers  xxxxxXXXXXXXXXXXXXXXXXXXXX
    train_filename = os.path.join(DIR_SAVE,'training_log.dat')
    val_filename = os.path.join(DIR_SAVE, 'validation_log.dat')

    logger_train = xlogger(filename=train_filename)
    logger_val = xlogger(filename=val_filename)
    # XXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxxxxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



    # Copy the source code in some destination directory, 
    # This creates a snapshot of all the files that where used for the run, for later inspection
    # Helps avoiding bugs 
    SRC_CODE_DIR = os.getcwd() # From where the source code was launched 
    flnames_dir = glob.glob(os.path.join(SRC_CODE_DIR,'*')) # All files and directories
    flnames_dir = list(filter(lambda x : '__pycache__' not in x, flnames_dir )) # Don't copy __pycache__ dirs
    flnames_dir = list(filter(lambda x : '.'  in x.split("/")[-1], flnames_dir )) # Exclude all directories from copying. 
    files_2_copy = list(filter(lambda x : 'slurm' not in x, flnames_dir )) # exclude slurm outputs

    # Create directory where source code snapshot is saved 
    dst_dir = os.path.join( DIR_SAVE,'source_code')
    construct_directory(DIR_SAVE,'source_code')

    # Create directory for saving weights: 
    WEIGHTS_DIR  = os.path.join(DIR_SAVE, 'model_weights')
    construct_directory(DIR_SAVE,'model_weights')

    # Copy operation from job launching directory to source code for logging
    for file in files_2_copy:
        # Alternative definition without shutil 
        os.system("cp -r {} {}".format(file,dst_dir)) 


    return DIR_SAVE, WEIGHTS_DIR, logger_train, logger_val



import pandas as pd 
import ast 
import numpy as np
def read_xlogfile(filename,sep="|",lineterminator='\n'):
    """
    Convenience function to read log files, the user must provide columns that correspond to lists
    """
    df = pd.read_csv(filename,sep=sep,lineterminator=lineterminator)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: np.array(ast.literal_eval(x)))
    return df


class xlogger(object):
    """
    Object to log dictionaries of values. 
    This object constructs files ending in .dat, and is used to log various variables during training. 
    This object DOES NOT track down if the dictionary provided for writing will change during the evolution of training
    (no reason to do so, right?).

    To be used with values that are floats or numpy arrays. 
    Use read_xlogfile to read the outputs of xlogger
    """
    def __init__(self, filename, ending='dat',sep='|',lineterminator='\n', append=False):
        """
        Default separator is "|", works well when storing mixed types, like floats and numpy arrays. 
        """
        self.f = filename
        self.sep = sep
        self.end = lineterminator
        if os.path.exists(self.f) and not append:
            print ("Warning, filename::{} exists, renaming to avoid overwriting".format(filename))
            head, tail = os.path.split(self.f)
            ending = tail.split('.')[-1] 

            timenow = datetime.now().strftime("%d-%m-%Y::%Hh-%Mm-%S")
            tail = tail.replace(ending,'_copy_on_{}.dat'.format(timenow))
            self.f = os.path.join(head,tail)
            print ("Logging in filename:{}".format(self.f))

    def write_helper(self,list_of_values, filename, open_mode):
        with open(filename,open_mode) as ff:
            print(*list_of_values, file=ff,flush=True,sep=self.sep,end=self.end)


    def write_header(self,kward):
        tlist = list(kward.keys())
        self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')
            
    def write(self,kward):
        # Trick to store numpy arrays into dictionary
        for k,v in kward.items():
            if isinstance(v,np.ndarray):
                kward[k]=v.tolist()
        if os.path.exists(self.f):
            tlist = kward.values()
            self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')
        else:
            self.write_header(kward)
            tlist = kward.values()
            self.write_helper(list_of_values=tlist,filename=self.f,open_mode='a')


# Convenience function to plot learning rate finder output
import matplotlib.pylab as plt
def plot_lr_finder(some_flname):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    df = read_xlogfile(some_flname)
    ax.plot(df.LearningRate,df.train_loss,'-.',label='training loss')
    ax.set_xscale('log')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('train loss')
    plt.legend()
    plt.show()


# TODO Implement option for model states 
def get_best_model_filename(flname_experiment,criterion='mcc',ascending=False, type='weights'):
    #flname_experiment = r'/scratch1/dia021/CloudMaskRSTeam/Results/FracTAL_D7nf32_on_22-05-2022::05h-29m-12s/'
    df = read_xlogfile(os.path.join(flname_experiment,'validation_log.dat'))
    epoch_load = df.sort_values(by=criterion,ascending=ascending).iloc[0].epoch

    dir_weights = os.path.join(flname_experiment,'model_weights')
    #dir_states = os.path.join(flname_experiment,'model_states')
    flname_weights = os.path.join(dir_weights,r'ModelWeights_epoch::{}.params'.format(epoch_load))
    #flname_states =  os.path.join(dir_states,r'ModelWeights_epoch::{}.params'.format(epoch_load))
    #print(flname_weights)
    
    return flname_weights