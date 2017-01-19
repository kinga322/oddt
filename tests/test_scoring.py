import os
from tempfile import NamedTemporaryFile

import numpy as np

from nose.tools import nottest
from nose.tools import assert_in, assert_not_in, assert_equal
from sklearn.utils.testing import (assert_true,
                                   assert_array_equal,
                                   assert_array_almost_equal)

import oddt
from oddt.scoring.functions import rfscore, nnscore

test_data_dir = os.path.dirname(os.path.abspath(__file__))


def test_rfscore():
    """Test RFScore v1-3 descriptors generators"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    rec.protein = True
    rec.addh()

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    for v in [1, 2, 3]:
        descs = rfscore(version=v, protein=rec).descriptor_generator.build(mols)
        # save correct results (for future use)
        # np.savetxt(os.path.join(test_data_dir,
        #                         'data/results/xiap/rfscore_v%i_descs.csv' % v),
        #            descs,
        #            fmt='%.16g',
        #            delimiter=',')
        descs_correct = np.loadtxt(os.path.join(test_data_dir, 'data/results/xiap/rfscore_v%i_descs.csv' % v), delimiter=',')

        # help debug errors
        for i in range(descs.shape[1]):
            mask = np.abs(descs[:, i] - descs_correct[:, i]) > 1e-4
            if mask.sum() > 1:
                print(i, np.vstack((descs[mask, i], descs_correct[mask, i])))

        assert_array_almost_equal(descs, descs_correct, decimal=4)


@nottest
def test_nnscore():
    """Test NNScore descriptors generators"""
    mols = list(oddt.toolkit.readfile('sdf', os.path.join(test_data_dir, 'data/dude/xiap/actives_docked.sdf')))
    list(map(lambda x: x.addh(), mols))

    rec = next(oddt.toolkit.readfile('pdb', os.path.join(test_data_dir, 'data/dude/xiap/receptor_rdkit.pdb')))
    rec.protein = True
    rec.addh()

    # Delete molecule which has differences in Acceptor-Donor def in RDK and OB
    del mols[65]

    # print((rec.atom_dict['atomicnum'] == 1).sum(),
    #       rec.atom_dict['isdonor'].sum(),
    #       rec.atom_dict['isdonorh'].sum())

    # for mol in mols:
    #     print(mol.num_rotors)
    #     print(sum(atom.Atom.GetAtomicNum() == 1 for atom in mol.atoms))
    #     print((mol.atom_dict['atomicnum'] == 1).sum(),
    #           mol.atom_dict['isdonor'].sum(),
    #           mol.atom_dict['isdonorh'].sum())

    gen = nnscore(protein=rec).descriptor_generator
    descs = gen.build(mols)
    # save correct results (for future use)
    # np.savetxt(os.path.join(test_data_dir,
    #                         'data/results/xiap/nnscore_descs.csv'),
    #            descs,
    #            fmt='%.16g',
    #            delimiter=',')
    descs_correct = np.loadtxt(os.path.join(test_data_dir, 'data/results/xiap/nnscore_descs.csv'), delimiter=',')

    # help debug errors
    for i in range(descs.shape[1]):
        mask = np.abs(descs[:, i] - descs_correct[:, i]) > 1e-4
        if mask.sum() > 1:
            print(i, gen.titles[i], mask.sum())
            print(np.vstack((descs[mask, i], descs_correct[mask, i])))

    assert_array_almost_equal(descs, descs_correct, decimal=4)