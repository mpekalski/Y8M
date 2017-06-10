# Copyright 2017 You8M team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from sklearn.model_selection import KFold
from tensorflow import gfile
import numpy as np

def split_fold(in_pattern, rettrain=True, fold=0, cvs=5, include_vlaidation=True, split_seed=0):
    """
    Splits the elements of the in_pattern into training and test sets
    :param in_pattern: string of tfrecord patterns
    :param rettrain: return training set (True) or leave out set (False)
    :param fold: which fold to process
    :param cvs: how many folds you want
    :param include_vlaidation: include validation set
    :return: subset of tfrecords
    """
    assert fold < cvs

    files = gfile.Glob(in_pattern)
    if split_seed > 0:
        kf = KFold(n_splits=cvs, shuffle=True, random_state=split_seed)
    else:
        kf = KFold(n_splits=cvs)

    for i, (train, test) in enumerate(kf.split(files)):
        if i == fold:
            break

    if rettrain:
        retfiles = list(np.array(files)[train])
    else:
        retfiles = list(np.array(files)[test])

    if include_vlaidation:
        addition = [fname.replace('train', 'validate') for fname in retfiles]
        retfiles += addition

    return retfiles
