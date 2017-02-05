#!/usr/bin/env python
#
#   Copyright 2017 Anil Thomas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Usage:
    ./run.py -w </path/to/data> -e 4 -r 0 -quick
"""
import os
import math
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from neon.util.compat import pickle
from neon.initializers import GlorotUniform
from neon.layers import GeneralizedCost, Affine, Conv
from neon.layers import DeepBiRNN, RecurrentMean, Dropout
from neon.models import Model
from neon.optimizers import Adadelta
from neon.transforms import SumSquared, Rectlin
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.data.datasets import Dataset
from neon.data.dataiterator import ArrayIterator


class Fin(Dataset):
    def __init__(self, nlags, path, quick):
        super(Fin, self).__init__(filename='train.h5', url=None,
                                  size=None, path=path)
        np.random.seed(0)
        self.nlags = nlags
        self.quick = quick
        self.load_data()
        self.shape = (1, self.nfeats, nlags)

    def load_data(self):
        data_file = os.path.join(
            self.path, 'findata-' + str(self.nlags) + '-' + str(self.quick) + '.pkl')
        if os.path.exists(data_file):
            print("Loading cached data from %s" % data_file)
            (self.nfeats, self.train_x, self.train_y,
             self.valid_x, self.valid_y) = pickle.load(file(data_file))
            return

        print("Processing data...")
        full = pd.read_hdf(os.path.join(self.path, self.filename), 'train')
        meds = full.median(axis=0)
        full.fillna(meds, inplace=True)
        cols = [col for col in full.columns if col not in ['id', 'timestamp', 'y']]
        self.nfeats = len(cols)

        uniq_ts = full['timestamp'].unique()
        mid = uniq_ts[len(uniq_ts)/2]
        train = full[full.timestamp < mid].reset_index()
        valid = full[full.timestamp >= mid].reset_index()

        if self.quick:
            train = train[train.id < 200].reset_index()
            valid = valid[valid.id < 200].reset_index()

        train_x, train_y = self.process(train, cols, self.nlags)
        valid_x, valid_y = self.process(valid, cols, self.nlags)
        self.train_x, self.train_y = self.shuffle(train_x, train_y)
        self.valid_x, self.valid_y = valid_x, valid_y
        pickle.dump((self.nfeats, self.train_x, self.train_y,
                     self.valid_x, self.valid_y), file(data_file, 'w'))
        print("Saved data to %s" % data_file)

    def shuffle(self, xs, ys):
        inds = np.arange(xs.shape[0])
        np.random.shuffle(inds)
        return xs[inds], ys[inds]

    def process(self, data, cols, nlags):
        """
        Returns features and targets. The feature set is expanded to include
        the specified number of time steps from the past.
        """
        ncols = self.nfeats
        xs = np.zeros((data.shape[0], nlags*ncols), dtype=np.float32)
        ys = np.zeros((data.shape[0], 1), dtype=np.float32)
        grouped = data.groupby('id')
        idx = 0
        # Group the samples according to "id" before collecting features from
        # previous time steps.
        for name, group in grouped:
            nrows = group.shape[0]
            inds = group.index.values
            if nrows < nlags:
                print('Warning: id %d has only %d samples' % (name, nrows))
            xgroup = np.zeros((nrows, nlags, ncols), dtype=np.float32)
            ygroup = np.zeros((nrows, 1), dtype=np.float32)
            xgroup[:, nlags-1] = group[cols].values
            ygroup[:] = group['y'].values.reshape((-1, 1))
            # Include features from previous time steps.
            for lag in range(1, min(nlags, nrows)):
                xgroup[:lag+1, nlags-lag-1] = group[cols].values[0]
                xgroup[lag:, nlags-lag-1] = group[cols].values[:-lag]
            # The transpose operation is to format the data as (N, F, T), where
            # N, F, T stand for batch, feature and time dimensions respectively.
            xs[idx:idx+nrows] = xgroup.transpose((0, 2, 1)).reshape((nrows, nlags*ncols))
            ys[idx:idx+nrows] = ygroup
            idx += nrows
        return xs, ys

    def gen_iterators(self):
        train = ArrayIterator(self.train_x, self.train_y, lshape=self.shape,
                              make_onehot=False, name='train')
        valid = ArrayIterator(self.valid_x, self.valid_y, lshape=self.shape,
                              make_onehot=False, name='valid')
        self._data_dict = {'train': train, 'valid': valid}
        return self._data_dict


def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return (np.sign(r2) * math.sqrt(math.fabs(r2)))

parser = NeonArgparser(__doc__)
parser.add_argument('-quick', '--quick_mode', action="store_true",
                    help="use a small subset of the data")
parser.add_argument('-ts', '--time_steps', default=7,
                    help='number of time steps')
args = parser.parse_args()
dataset = Fin(nlags=int(args.time_steps), path=args.data_dir, quick=args.quick_mode)

train = dataset.train_iter
valid = dataset.valid_iter

init = GlorotUniform()
opt = Adadelta()
common = dict(init=init, dilation=dict(dil_h=2, dil_w=2),
              padding=dict(pad_h=0, pad_w=1), activation=Rectlin(),
              batch_norm=True)
nchan = 16
layers = [Conv((1, 2, nchan), **common),
          Conv((1, 2, nchan), **common),
          Conv((1, 2, nchan/2), **common),
          Conv((1, 2, nchan/4), **common),
          Conv((1, 2, nchan/8), **common),
          Conv((1, 2, nchan/16), **common),
          Dropout(0.8),
          DeepBiRNN(16, init=init, activation=Rectlin(), reset_cells=True, depth=3),
          RecurrentMean(),
          Affine(nout=1, init=init, activation=None)]

cost = GeneralizedCost(costfunc=SumSquared())
net = Model(layers=layers)
callbacks = Callbacks(net, eval_set=valid, **args.callback_args)

net.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

train_preds = net.get_outputs(train)
print('  training R %.4f' % r_score(dataset.train_y, train_preds))
valid_preds = net.get_outputs(valid)
print('validation R %.4f' % r_score(dataset.valid_y, valid_preds))
