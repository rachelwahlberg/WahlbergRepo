
from pathlib import Path
import pandas as pd
import re
import numpy as np
from neuropy.io.openephysio import create_sync_df

def align_events(filepath:str or Path, labels:list = None, sanity_check = False):
    
    """ bring in txt file (from national instruments, for ex and align times with dat file
    
    ------ params/assumptions --------- 
    
    send in column_names as ['time','port'] etc. make sure in order
    assumes first col is times. can have extra cols after that as long as there's a label per col.
    assumes file is inside a folder inside the basepath.
    
    ----- ex usage -------
    port_times = align_timestamps(port_timestamps_file[0],column_names=['times','port'])
    """
    ## ---------- initial import ----------------
    with open(filepath, 'r') as f:
       # cols = list(zip(*[line.strip().split() for line in f]))
        lines = [line.strip().split() for line in f]
        max_cols = max(len(line) for line in lines)# Find the maximum number of columns
        # Pad shorter rows with None (or any default value)
        padded_lines = [line + [None] * (max_cols - len(line)) for line in lines]
        cols = list(zip(*padded_lines))



    if len(cols) != len(labels):
        raise ValueError("Must have equal number of labels and data columns.")
    
    times = cols[0]

    #----- rachel specific -----

    if labels[1] == 'port':
        ports = np.array(list(map(int, cols[1])))
        ports = ports - 1 #### DELETE AFTER CHO/PETUNIA
        cols[1]=ports

    if labels[1] == 'port1':
        port1 = np.array([int(x) if x is not None else None for x in cols[1]], dtype=object)
        #port1 = np.array([int(x) if x is not None else np.nan for x in cols[1]])
        cols[1]=port1# for block file, doesn't need the -1

    if (len(labels)>2):
        if labels[2] == 'port2':
            port2 = np.array([int(x) if x is not None else None for x in cols[2]], dtype=object)

            #port2 = np.array([int(x) if x is not None else np.nan for x in cols[2]])
            cols[2]=port2

    ##-------------- convert to datetime ---------
    filedate = re.search(r'(\d{4}-\d{2}-\d{2})', times[0])
    if filedate == None: # add filedate if it was not included in the time within the file
        filedate = re.search(r'(\d{4}-\d{2}-\d{2})', filepath.name)
        assert filedate != None,"add filedate to file name in format yyyy-mm-dd"
        filedate = filedate.group(1)
        times= [f"{filedate} {time}" for time in times]

    #convert to datetime
    datetimes = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S:%f')
    datetimes = datetimes.tz_localize('America/Detroit')

    # --------------- sync with eeg recording -------------
    # Get sync data
    basepath = filepath.parent.parent #assumes your file is in a folder within the basepath!
   # start_end_df = get_dat_timestamps(basepath, start_end_only=True, print_start_time_to_screen=False)
    sync_df = create_sync_df(basepath)

    timedif = []
    all_event_times = []
    for rec in sync_df.Recording.unique():
        rec_start = sync_df[(sync_df['Recording'] == rec) & 
                        (sync_df['Condition'] == 'start')]['Datetime'].iloc[0]
        rec_stop = sync_df[(sync_df['Recording'] == rec) & 
                        (sync_df['Condition'] == 'stop')]['Datetime'].iloc[0]
        rec_eeg_starttime = sync_df[(sync_df['Recording'] == rec) & 
                        (sync_df['Condition'] == 'start')]['eeg_time'].iloc[0]


        use_bool = (datetimes > rec_start) & (datetimes < rec_stop) #times during this file
        timedif = (datetimes[use_bool] - rec_start).total_seconds() #get time from filestart
        
        event_times=[t + rec_eeg_starttime for t in timedif]
        all_event_times.extend(event_times)

        timedif = []

    if len(cols[0])!=len(all_event_times): 
        all_event_times.extend([sync_df.eeg_time.iloc[-1]])
        cols[0] = all_event_times
        print('used sync_df.eeg_time.iloc[-1] for recording end time - was left running')
    else:
        cols[0] = all_event_times #if this errors,check if a time was outside of all the recording times.
    
    d = dict(zip(labels, cols))# Create a dictionary with the specified column names
    eventData = pd.DataFrame(d)

    return eventData

    