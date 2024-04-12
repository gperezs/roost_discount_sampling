import os
import sys
from os.path import join
import glob
import argparse
import pandas as pd 

sys.path.insert(0, 'src')
from data_carp import *
from discount import kDISCount

import warnings
warnings.filterwarnings("ignore") # there is an annoying warning from pandas


def parse_args():
    parser = argparse.ArgumentParser(description='Sample days from roost detector counts with discount')
    parser.add_argument('--data-dir', default='data/us_sunrise_v3_pilot0124', type=str, 
                        help='directory with sweeps and tracks files')
    parser.add_argument('--out-dir', default='sampled_days_files', type=str,
                        help='directory to save txt files with sampled days')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sweeps_files = glob.glob(join(args.data_dir, 'sweeps_*'))
    sweeps_files = [filename.split('/')[-1] for filename in sweeps_files]

    for i, sweeps_path in enumerate(sweeps_files):

        station = sweeps_path.split('_')[1] # get station from filename
        year = sweeps_path.split('_')[2][:4] # get year from filename

        # Load files
        tracks_path = 'tracks_' + '_'.join(sweeps_path.split('_')[1:])

        sweeps_df = pd.read_csv(join(args.data_dir, sweeps_path))
        tracks_df = pd.read_csv(join(args.data_dir, tracks_path))

        df = sweeps_df.merge(tracks_df)

        df["date"] = df.apply(lambda x: x.local_time[0:8], axis = 1)
        df['track_id'] = df.apply(lambda x: x.filename[0:4] + x.date + "-" + str(x.track_id), axis = 1)
        df["detection_id"] = df.apply(lambda x: x.track_id + x.filename[12:], axis = 1)

        df = summarize_sweeps(df)
        df = create_det_idx(df)

        df = summarize_tracks(df)

        detector_counts = df.groupby("date").n_animals.sum().tolist() #g(s)

        print('%s %s (%d days)'%(station, year, len(detector_counts)))

        # Get list of days (for text file)
        days = []
        for filename in df.drop_duplicates('filename')['filename']:
            date = filename.split('_')[0][4:]
            if date not in days:
                days.append(date)

        # Load discount and sample
        estimator = kDISCount(detector_counts) # initialize estimator with g(s)
        samples = estimator.sample(n=len(detector_counts))

        samples = [days[i] for i in samples] # sampling ids to dates

        # save text file with sampled days
        with open(join(args.out_dir, 'discount_%s%s.txt'%(station, year)), mode='w') as f:
            f.write('\n'.join(samples))
            f.close()
    print('done.')


