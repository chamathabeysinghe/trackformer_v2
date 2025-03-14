# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .mot17_sequence import MOT17Sequence
from .mots20_sequence import MOTS20Sequence
from .ant_sequence import AntSequence


class MOT17Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        train_sequences = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = [
            'MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
            'MOT17-08', 'MOT17-12', 'MOT17-14']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            if dets == 'ALL':
                self._data.append(MOT17Sequence(seq_name=seq, dets='DPM', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='FRCNN', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='SDP', **kwargs))
            else:
                self._data.append(MOT17Sequence(seq_name=seq, dets=dets, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOTS20Wrapper(MOT17Wrapper):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOTS20Sequence dataset
        """
        train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOTS20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOTS20-{split}"]
        else:
            raise NotImplementedError("MOTS20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOTS20Sequence(seq_name=seq, **kwargs))

class AntWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        # train_sequences = ["OU10B3L2In_0", "OU10B1L1Out_0", "OU10B3L3In_0", "OU10B2L2In_0", "OU10B2L3Out_0",
        #                    "OU10B2L2Out_0", "OU10B2L3In_0", "OU10B1L2In_0", "OU10B1L3Out_0", "OU10B1L2Out_0",
        #                    "OU10B1L3In_0", "OU50B1L1In_0", "OU50B1L1Out_0", "OU50B1L2In_0", "OU50B2L2Out_0",
        #                    "OU50B1L3In_0", "OU50B1L3Out_0", "OU50B1L2Out_0", "OU50B2L2In_0", "OU50B2L3In_0"]
        train_sequences = ['CU10L1B1In_0', 'CU10L1B1Out_0', 'CU25L1B1Out_0', 'CU25L1B1In_0',
             'CU15L1B4In_0', 'CU15L1B4Out_0', 'CU20L1B4In_0', 'CU20L1B4Out_0',
             'CU50L1B6In_0', 'CU50L1B6Out_0',
             'CU10L1B5In_0', 'CU10L1B6Out_0', 'CU15L1B1In_0', 'CU20L1B1Out_0', 'CU10L1B4In_0', 'CU25L1B4In_0', 'CU10L1B6In_0', 'CU10L1B5Out_0', 'CU30L1B6Out_1', 'CU30L1B6Out_2', 'CU30L1B6Out_3',
                           'OU10B1L1Out_0_mod_0', 'OU10B1L2Out_0_mod_0', 'OU10B1L3Out_0_mod_0', 'OU10B1L1Out_0_mod_1',
                           'OU10B1L2Out_0_mod_1', 'OU10B1L3Out_0_mod_1','colony3_small_day_2', 'colony5_small_day_2', 'colony6_small_day_1', 'colony7_small_day_1', "colony3_large_day_1", "colony5_large_day1", "colony6_large_day_2", "colony7_large_day_2",
                           ]
        # test_sequences = ["OU10B3L2Out_0", "OU10B3L3Out_0", "OU10B1L1In_0", "OU50B3L2In_0", "OU50B3L2Out_0", "OU50B3L3Out_0"]
        test_sequences = ['CU15L1B1Out_0', 'CU20L1B1In_0', 'CU10L1B4Out_0', 'CU25L1B4Out_0', 'CU30L1B6In_0', 'CU30L1B6Out_0',
                          'OU10B1L1Out_0_mod_0', 'OU10B1L1Out_0_mod_1','colony3_small_day_2', 'colony5_small_day_2', 'colony6_small_day_1', 'colony7_small_day_1']
        all_sequences = ["CU10L1B1In_0", "CU10L1B1Out_0", "CU25L1B1Out_0", "CU25L1B1In_0", "CU10L1B4In_0", "CU10L1B4Out_0",
              "CU25L1B4In_0", "CU25L1B4Out_0", "CU10L1B5In_0", "CU10L1B6Out_0", "CU50L1B6In_0", "CU50L1B6Out_0",
              "CU15L1B1Out_0", "CU20L1B1In_0", "CU15L1B4Out_0", "CU20L1B4In_0", "CU10L1B5Out_0", "CU30L1B6In_0",
              "CU15L1B1In_0", "CU20L1B1Out_0", "CU15L1B4In_0", "CU20L1B4Out_0", "CU30L1B6Out_0", "CU10L1B6In_0",
              "OU10B1L1Out_0", "OU50B1L1In_0", "OU10B1L2Out_0", "OU50B1L2Out_0", "OU10B1L3In_0", "OU50B1L3In_0",
              "OU10B2L1Out_0", "OU50B2L1In_0", "OU10B2L2Out_0", "OU50B2L2In_0", "OU10B2L3In_0", "OU10B2L3Out_0",
              "OU50B2L3In_0", "OU50B3L1In_0", "OU10B3L2In_0", "OU10B3L2Out_0", "OU50B3L2In_0", "OU10B3L3In_0",
              "OU10B3L3Out_0", "OU50B3L3Out_0", "OU10B1L1In_0", "OU50B1L2In_0", "OU50B1L3Out_0", "OU10B2L2In_0",
              "OU10B3L1Out_0", "OU50B3L3In_0", "OU50B1L1Out_0", "OU10B1L2In_0", "OU10B1L3Out_0", "OU10B2L1In_0",
              "OU50B2L2Out_0", "OU10B3L1In_0", "OU50B3L2Out_0",
              'OU10B1L1Out_0_mod_0', 'OU10B1L2Out_0_mod_0', 'OU10B1L3Out_0_mod_0', 'OU10B1L1Out_0_mod_1',
              'OU10B1L2Out_0_mod_1', 'OU10B1L3Out_0_mod_1',
             'colony3_small_day_2', 'colony5_small_day_2', 'colony6_small_day_1', 'colony7_small_day_1',
             "colony3_large_day_1", "colony5_large_day1", "colony6_large_day_2", "colony7_large_day_2",
             'task_switching_validation', 'task_switching_test',
             "P1320451", "P1320462", "P1320470", "P1320480", "P1320488", "P1320559_1", "P1320578",
             "P1330020",
             "P1320453", "P1320464", "P1320473", "P1320482", "P1320489", "P1320563", "P1320587", "P1330028",
             "P1320459", "P1320467", "P1320476", "P1320484", "P1320551", "P1320565", "P1320592",
             "P1320460", "P1320469", "P1320478", "P1320486", "P1320554", "P1320573", "P1320783",
                         'P1320472', 'P1320466', 'P1320475', 'P1320465', 'P1320463', 'P1320477', 'P1320575', 'P1320788',
                         'P1320474', 'P1320592_1', 'P1320458', 'P1330029', 'P1320789', 'P1320461', 'P1320471',
                         'P1320561'

                         ]
        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif split in all_sequences:
            sequences = [split]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(AntSequence(seq_name=seq, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

