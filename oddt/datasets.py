""" Datasets wrapped in convenient models """
from os.path import isfile, isdir
from os import listdir
import csv
import pandas as pd
import six

from oddt import toolkit


class pdbbind(object):
    def __init__(self,
                 home,
                 version=None,
                 default_set=None,
                 data_file=None,
                 opt=None):
        version = int(version)
        self.home = home
        if default_set:
            self.default_set = default_set
        else:
            if version == 2007:
                self.default_set = 'general'
            else:
                self.default_set = 'general_PL'
        self.opt = opt or {}
        self.sets = {}
        self._set_ids = {}
        self._set_act = {}

        if version:
            if version == 2007:
                pdbind_sets = ['core', 'refined', 'general']
            else:
                pdbind_sets = ['core', 'refined', 'general_PL']
            for pdbind_set in pdbind_sets:
                if data_file:
                    csv_file = data_file
                elif version == 2007:
                    csv_file = os.path.join(self.home,
                                            'INDEX.%i.%s.data' % (version, pdbind_set))
                elif version == 2016:
                    csv_file = os.path.join(self.home,
                                            'index',
                                            'INDEX_%s_data.%i' % (pdbind_set, version))
                else:
                    csv_file = os.path.join(self.home,
                                            'INDEX_%s_data.%i' % (pdbind_set, version))

                if os.path.isfile(csv_file):
                    data = pd.read_csv(csv_file,
                                       sep='\s+',
                                       usecols=[0, 1, 2, 3],
                                       names=['pdbid',
                                              'resolution',
                                              'release_year',
                                              'act'],
                                       comment='#')
                    self._set_ids[pdbind_set] = data['pdbid'].tolist()
                    self._set_act[pdbind_set] = data['act'].tolist()
                    self.sets[pdbind_set] = dict(zip(self._set_ids[pdbind_set],
                                                     self._set_act[pdbind_set]))
            if len(self.sets) == 0:
                raise Exception('There is no PDBbind set availabe')
        else:
            pass  # list directory, but no metadata then

    @property
    def ids(self):
        # return sorted(self.sets[self.default_set].keys())
        return self._set_ids[self.default_set]

    @property
    def activities(self):
        return self._set_act[self.default_set]

    def __iter__(self):
        for pdbid in self.ids:
            yield _pdbbind_id(self.home, pdbid, opt=self.opt)

    def __getitem__(self, pdbid):
        if pdbid in self.ids:
            return _pdbbind_id(self.home, pdbid, opt=self.opt)
        else:
            if type(pdbid) is int:
                return _pdbbind_id(self.home + '', self.ids[pdbid], opt=self.opt)
            return None


class _pdbbind_id(object):
    def __init__(self, home, pdbid, opt=None):
        self.home = home
        self.id = pdbid
        self.opt = opt or {}

    @property
    def protein(self):
        f = os.path.join(self.home, self.id, '%s_protein.pdb' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def pocket(self):
        f = os.path.join(self.home, self.id, '%s_pocket.pdb' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
        else:
            return None

    @property
    def ligand(self):
        f = os.path.join(self.home, self.id, '%s_ligand.sdf' % self.id)
        if os.path.isfile(f):
            return next(toolkit.readfile('sdf', f, lazy=True, opt=self.opt))
        else:
            return None


class dude(object):

    def __init__(self, home):
        """A wrapper for DUD-E (A Database of Useful Decoys: Enhanced)
        http://dude.docking.org/

        Parameters
        ----------
        home : str
            Path to files from dud-e

        """
        self.home = home
        if not os.path.isdir(self.home):
            raise Exception('Directory %s doesn\'t exist' % self.home)

        self.ids = []
        files = ['receptor.pdb', 'crystal_ligand.mol2', 'actives_final.mol2.gz', 'decoys_final.mol2.gz']
        # ids sorted by size of protein
        all_ids = ['fnta', 'dpp4', 'mmp13', 'hivpr', 'ada17', 'mk14', 'egfr', 'src', 'drd3', 'aa2ar',
                   'cah2', 'parp1', 'cdk2', 'lck', 'pde5a', 'thrb', 'aces', 'try1', 'pparg', 'vgfr2',
                   'pgh2', 'esr1', 'fa10', 'esr2', 'ppara', 'dhi1', 'hivrt', 'bace1', 'ace', 'dyr',
                   'akt1', 'adrb1', 'prgr', 'gcr', 'adrb2', 'andr', 'ppard', 'csf1r', 'gria2', 'cp3a4',
                   'met', 'pgh1', 'abl1', 'casp3', 'kit', 'hdac8', 'hdac2', 'braf', 'urok', 'lkha4',
                   'igf1r', 'aldr', 'fpps', 'hmdh', 'kpcb', 'tgfr1', 'ital', 'mp2k1', 'nos1', 'tryb1',
                   'rxra', 'thb', 'cp2c9', 'ptn1', 'reni', 'pnph', 'tysy', 'akt2', 'kif11', 'aofb',
                   'plk1', 'hivint', 'mk10', 'pyrd', 'grik1', 'jak2', 'rock1', 'fa7', 'mapk2', 'nram',
                   'wee1', 'fkb1a', 'def', 'ada', 'fak1', 'mcr', 'pa2ga', 'xiap', 'hs90a', 'hxk4',
                   'mk01', 'pygm', 'glcm', 'comt', 'sahh', 'cxcr4', 'kith', 'ampc', 'pur2', 'fabp4',
                   'inha', 'fgfr1']
        for i in all_ids:
            if os.path.isdir(os.path.join(self.home, i)):
                self.ids.append(i)
                for file in files:
                    f = os.path.join(self.home, i, file)
                    if not os.path.isfile(f) and not (file[-3:] == '.gz' and os.path.isfile(f[:-3])):
                        print('Target %s doesn\'t have file %s' % (i, file), file=sys.stderr)
        if not self.ids:
            print('No targets in directory %s' % (self.home), file=sys.stderr)

    def __iter__(self):
        for dude_id in self.ids:
            yield _dude_target(self.home, dude_id)

    def __getitem__(self, dude_id):
        if dude_id in self.ids:
            return _dude_target(self.home, dude_id)
        else:
            raise Exception('Directory %s doesn\'t exist' % self.home)


class _dude_target(object):

    def __init__(self, home, dude_id):
        """Allows to read files of the dude target

        Parameters
        ----------
        home : str
            Directory to files from dud-e

        dude_id : str
            Target id
        """
        self.home = home
        self.dude_id = dude_id

    @property
    def protein(self):
        """Read a protein file"""
        f = os.path.join(self.home, self.dude_id, 'receptor.pdb')
        if os.path.isfile(f):
            return next(toolkit.readfile('pdb', f))
        else:
            return None

    @property
    def ligand(self):
        """Read a ligand file"""
        f = os.path.join(self.home, self.dude_id, 'crystal_ligand.mol2')
        if os.path.isfile(f):
            return next(toolkit.readfile('mol2', f))
        else:
            return None

    @property
    def actives(self):
        """Read an actives file"""
        f = os.path.join(self.home, self.dude_id, 'actives_final.mol2.gz')
        if os.path.isfile(f):
            return toolkit.readfile('mol2', f)
        # check if file is unpacked
        elif os.path.isfile(f[:-3]):
            return toolkit.readfile('mol2', f[:-3])
        else:
            return None

    @property
    def decoys(self):
        """Read a decoys file"""
        f = os.path.join(self.home, self.dude_id, 'decoys_final.mol2.gz')
        if os.path.isfile(f):
            return toolkit.readfile('mol2', f)
        # check if file is unpacked
        elif os.path.isfile(f[:-3]):
            return toolkit.readfile('mol2', f[:-3])
        else:
            return None


class CASF:
    def __init__(self, home):
        self.home = home
        self.index = '%s/coreset/index/' % self.home

        if isdir(self.index):
            filepath = '%s/2013_core_data.lst' % self.index
            self.index_data = pd.read_csv(filepath, sep='\s+', comment='#', header=None,
                                          names=['pdbid', 'act', 'cluster'],  usecols=[0, 1, 5])
            self.pdbids = self.index_data['pdbid']

    def __iter__(self):
        for pdbid in self.pdbids:
            yield _CASFTarget(self.home, pdbid)

    def __getitem__(self, item):
        if item in self.pdbids:
            return _CASFTarget(self.home, item)
        elif isinstance(int, item) and item < len(self.pdbids):
            return _CASFTarget(self.home, self.pdbids[item])
        else:
            raise KeyError

    def precomputed_score(self, scoring_function=None):
        examples_dir = '%s/power_scoring/examples' % self.home
        if scoring_function is not None:
            functions = [scoring_function]
        else:
            functions = listdir(examples_dir)
            functions.remove('README')

        frames = []

        for fun in functions:
            file_score = '%s/%s' % (examples_dir, fun)
            if not isfile(file_score):
                raise FileNotFoundError('Invalid scoring function name')

            score = pd.read_csv(file_score, comment='#', sep='\s+', header=None,
                                names=['pdbid', 'score_crystal', 'score_opt'])
            act = self.index_data[['pdbid', 'act']]

            scores = pd.merge(score, act)
            scores['scoring_function'] = pd.Series([fun] * 195, name='Scoring function')
            frames.append(scores)

        return pd.concat(frames)

    def precomputed_screening(self, scoring_function=None, cluster_id=None):
        screening_dir = '%s/power_screening' % self.home
        examples_dir = '%s/examples' % screening_dir
        if scoring_function is not None:
            functions = [scoring_function]
        else:
            functions = listdir(examples_dir)

        cluster_frame = pd.DataFrame(columns=['cluster_id', 'protein_structure', 'cluster_proteins'])
        data_file = open('%s/TargetInfo.dat' % screening_dir)

        for cluster, line in enumerate(filter(lambda x: not x.startswith('#'), data_file.readlines())):
            line = line.split()
            protein_structure = line[0]
            cluster_proteins = line[1:]
            cluster_frame.loc[cluster] = [cluster + 1, protein_structure, cluster_proteins]

        frames = []
        for fun in functions:
            file_dir = '%s/%s' % (examples_dir, fun)
            if not isdir(file_dir):
                raise FileNotFoundError('Invalid scoring function name')
            if cluster_id:
                protein = cluster_frame.iloc[cluster_id - 1]['protein_structure']
                frame = pd.read_csv('%s/%s_score.dat' % (file_dir, protein), sep='\s+',
                                    header=None, names=['name', 'score'])
                frame['pdbid'] = [name[:4] for name in frame['name']]
                frame['scoring_function'] = [fun] * len(frame)
                frame = frame.merge(self.index_data[['pdbid', 'act']])
                frames.append(frame)

            else:
                for row in cluster_frame.itertuples():
                    protein = row[2]
                    frame = pd.read_csv('%s/%s_score.dat' % (file_dir, protein), sep='\s+',
                                        header=None, names=['name', 'score'])
                    x = row[1]
                    frame['cluster_id'] = [x] * len(frame)
                    frame['protein_structure'] = [protein] * len(frame)
                    frame['cluster_proteins'] = [row[3]] * len(frame)
                    frame['pdbid'] = [name[:4] for name in frame['name']]
                    frame['scoring_function'] = [fun] * len(frame)
                    frame = frame.merge(self.index_data[['pdbid', 'act']])
                    frames.append(frame)

        return pd.concat(frames, ignore_index=True)


class _CASFTarget:
    def __init__(self, home, pdbid):
        self.home = home
        self.pdbid = pdbid

    @property
    def protein(self):
        filepath = '%s/coreset/%s/%s_protein.mol2' % (
            self.home, self.pdbid, self.pdbid)
        if isfile(filepath):
            protein = six.next(toolkit.readfile('mol2', filepath))
            return protein
        return None

    @property
    def ligand(self):
        filepath = '%s/coreset/%s/%s_ligand.mol2' % (
            self.home, self.pdbid, self.pdbid)
        if isfile(filepath):
            ligand = six.next(toolkit.readfile('mol2', filepath))
            return ligand
        return None

    @property
    def decoys_docking(self):
        filepath = '%s/decoys_docking/%s_decoys.mol2' % (self.home, self.pdbid)
        if isfile(filepath):
            decoys = list(toolkit.readfile('mol2', filepath))
            return decoys
        return None

    @property
    def decoys_screening(self):
        dirpath = '%s/decoys_screening/%s' % (self.home, self.pdbid)
        if isdir(dirpath):
            decoys = []
            for file in listdir(dirpath):
                decoys.append(six.next(toolkit.readfile('mol2', dirpath + '/' + file)))
            return decoys
        return None
