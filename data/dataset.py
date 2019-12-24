from itertools import zip_longest
import numpy as np
import tables


class ZINCH5Dataset(object):
    def __init__(self, arrayh5=(), fph5=()):
        self.arrayh5_r = [tables.open_file(f, 'r').root for f in arrayh5]
        self.fph5_r = [tables.open_file(f, 'r').root for f in fph5]
        assert len(self.arrayh5_r) == len(self.fph5_r)
        self.data_sample = None

    def __getitem__(self, item):
        # self[tranche_id, line_id]
        # return [mol_idx, mol_array, mol_fp]
        assert isinstance(item, (tuple, np.ndarray)) and len(item) == 2
        trch, line = int(item[0]), int(item[1])
        return (
            self.arrayh5_r[trch].mol_array[line],
            self.fph5_r[trch].mol_fp[line],
            self.arrayh5_r[trch].mol_idx[line],
        )

    def sample(self, num_samp=None, train_ratio=0.9, random_seed=1999):
        np.random.seed(random_seed)
        if num_samp:
            num_samp_ = int(num_samp / len(self.arrayh5_r))
            num_samp_ = [num_samp_, ] * (len(self.arrayh5_r) - 1) + [num_samp - (len(self.arrayh5_r) - 1) * num_samp_, ]
            samp = [np.random.permutation(len(r.mol_idx))[:n] for r, n in zip(self.arrayh5_r, num_samp_)]
        else:
            samp = [np.random.permutation(len(r.mol_idx)) for r in self.arrayh5_r]
            num_samp = sum([len(r.mol_idx) for r in self.arrayh5_r])
        samp = [list(map(lambda x: (i, x), s)) for i, s in enumerate(samp)]
        samp = np.random.permutation([j for i in samp for j in i])
        self.data_sample = {
            'train': samp[:int(len(samp) * train_ratio)],
            'test': samp[int(len(samp) * train_ratio):],
        }
        return num_samp


class ZINCH5Dataloader(object):
    def __init__(self, zinc=None, batch_size=None, num_workers=5,):
        self.dataset = zinc
        self.num_workers = num_workers
        self.batch_size = batch_size

    @staticmethod
    def restore_flat_array(fa):
        fa = fa.reshape(-1, 5)
        return (fa[:, 0].tolist(),
                fa[:, 1].astype('int').tolist(),
                fa[:, 2:].tolist())

    def _get_batch(self, data):
        if data[-1]:
            idxs, fps = [], []
            nums_atoms, gs_charge, atom_type, pos = [], [], [], []
            for array, fp, idx in data:
                array = self.restore_flat_array(array)
                nums_atoms.append(len(array[0]))
                gs_charge += array[0]
                atom_type += array[1]
                pos += array[2]
                fps.append(fp.tolist())  # array to list
                idxs.append(int(idx))  # np.int to int
            return ((gs_charge, atom_type, pos, nums_atoms), fps), idxs
        else:
            return None

    def __call__(self, phase):
        '''

        Parameters
        ----------
        phase: str
            'train' or 'test'

        Returns
        -------
        data:
            batch_size * [(array, fps), idxs], len(nums_atoms) === len(fps) === len(idxs)
        '''
        if self.dataset.sample:
            sample2load = map(self.dataset.__getitem__, self.dataset.data_sample[phase])
        else:
            raise ValueError('No Sample.')
        batched_set = zip_longest(*[sample2load, ] * self.batch_size)
        for data in map(self._get_batch, batched_set):
            # the last batch discarded
            if data:
                yield data
