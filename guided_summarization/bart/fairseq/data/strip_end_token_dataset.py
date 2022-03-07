# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class StripEndTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return item[:-1]
