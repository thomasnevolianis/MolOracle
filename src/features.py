from rdkit.Chem.rdchem import Atom, Bond, BondType, HybridizationType, BondStereo, ChiralType
import copy
import torch

# specify custom functions
# rdkit functions which cannot simply be called with <function_name>(atom/bond) for obtaining feature values
def atom_GetNeighbors(atom):
    return len(Atom.GetNeighbors(atom))

def atom_GetTotalNumHs(atom):
    return Atom.GetTotalNumHs(atom, includeNeighbors=True)

feature_dict = {
    "atom": {
        "type": {
            "mapping_dict": {
                'H': 1,
                'He': 2,
                'Li': 3,
                'Be': 4,
                'B': 5,
                'C': 6,
                'N': 7,
                'O': 8,
                'F': 9,
                'Ne': 10,
                'Na': 11,
                'Mg': 12,
                'Al': 13,
                'Si': 14,
                'P': 15,
                'S': 16,
                'Cl': 17,
                'Ar': 18,
                'K': 19,
                'Ca': 20,
                'Sc': 21,
                'Ti': 22,
                'V': 23,
                'Cr': 24,
                'Mn': 25,
                'Fe': 26,
                'Co': 27,
                'Ni': 28,
                'Cu': 29,
                'Zn': 30,
                'Ga': 31,
                'Ge': 32,
                'As': 33,
                'Se': 34,
                'Br': 35,
                'Kr': 36,
                'Rb': 37,
                'Sr': 38,
                'Y': 39,
                'Zr': 40,
                'Nb': 41,
                'Mo': 42,
                'Tc': 43,
                'Ru': 44,
                'Rh': 45,
                'Pd': 46,
                'Ag': 47,
                'Cd': 48,
                'In': 49,
                'Sn': 50,
                'Sb': 51,
                'Te': 52,
                'I': 53,
                'Xe': 54,
                'Cs': 55,
                'Ba': 56,
                'La': 57,
                'Ce': 58,
                'Pr': 59,
                'Nd': 60,
                'Pm': 61,
                'Sm': 62,
                'Eu': 63,
                'Gd': 64,
                'Tb': 65,
                'Dy': 66,
                'Ho': 67,
                'Er': 68,
                'Tm': 69,
                'Yb': 70,
                'Lu': 71,
                'Hf': 72,
                'Ta': 73,
                'W': 74,
                'Re': 75,
                'Os': 76,
                'Ir': 77,
                'Pt': 78,
                'Au': 79,
                'Hg': 80,
                'Tl': 81,
                'Pb': 82,
                'Bi': 83,
                'Po': 84,
                'At': 85,
                'Rn': 86,
                'Fr': 87,
                'Ra': 88,
                'Ac': 89,
                'Th': 90,
                'Pa': 91,
                'U': 92,
                'Np': 93,
                'Pu': 94,
                'Am': 95,
                'Cm': 96,
                'Bk': 97,
                'Cf': 98,
                'Es': 99,
                'Fm': 100,
                'Md': 101,
                'No': 102,
                'Lr': 103,
                'Rf': 104,
                'Db': 105,
                'Sg': 106,
                'Bh': 107,
                'Hs': 108,
                'Mt': 109,
                'Ds': 110,
                'Rg': 111,
                'Cn': 112,
                'Uut': 113,
                'Fl': 114,
                'Uup': 115,
                'Lv': 116,
                'Uus': 117,
                'Uuo': 118,
            },
            "rdkit_func": Atom.GetSymbol
        },
        "aromatic_1class": {
            "mapping_dict": {
                False: 0,
                True: 1
            },
            "rdkit_func": Atom.GetIsAromatic
        },
        "ring_1class": {
            "mapping_dict": {
                False: 0,
                True: 1
            },
            "rdkit_func": Atom.IsInRing
        },
        "hybridization": {
            "mapping_dict": {
                HybridizationType.UNSPECIFIED: 0,
                HybridizationType.S: 1,
                HybridizationType.SP: 2,
                HybridizationType.SP2: 3,
                HybridizationType.SP3: 4,
                #HybridizationType.SP2D: 5, # used according to RDKit website, but not available
                HybridizationType.SP3D: 6,
                HybridizationType.SP3D2: 7,
                HybridizationType.OTHER: 8
            },
            "rdkit_func": Atom.GetHybridization
        },
        "formal_charge": {
            "mapping_dict": {
                -4: 0, # note that torch requires non-negative values for one-hot encoding
                -3: 1,
                -2: 2,
                -1: 3,
                0: 4,
                1: 5,
                2: 6,
                3: 7,
                4: 8,
                5: 9,
                6: 10,
                7: 11,
                8: 12,
            },
            "rdkit_func": Atom.GetFormalCharge
        },
        "num_hs": {
            "mapping_dict": {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
            },
            "rdkit_func": atom_GetTotalNumHs
        },
        "num_neighbors": {
            "mapping_dict": {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
            },
            "rdkit_func": atom_GetNeighbors
        },
        "chirality": {
            "mapping_dict": {
                ChiralType.CHI_UNSPECIFIED: 0,
                ChiralType.CHI_TETRAHEDRAL_CW: 1,
                ChiralType.CHI_TETRAHEDRAL_CCW: 2,
                #ChiralType.CHI_TETRAHEDRAL: 3, # used according to RDKit website, but not available
                #ChiralType.CHI_ALLENE: 4, # used according to RDKit website, but not available
                #ChiralType.CHI_SQUAREPLANAR: 5, # used according to RDKit website, but not available
                #ChiralType.CHI_TRIGONALBIPYRAMIDAL: 6, # used according to RDKit website, but not available
                #ChiralType.CHI_OCTAHEDRAL: 7, # used according to RDKit website, but not available
                ChiralType.CHI_OTHER: 8
            },  
            "rdkit_func": Atom.GetChiralTag
        }
    },
    "bond": {
        "type": {
            "mapping_dict": {
                BondType.SINGLE: 0, 
                BondType.DOUBLE: 1, 
                BondType.TRIPLE: 2, 
                BondType.AROMATIC: 3
            },
            "rdkit_func": Bond.GetBondType
        },
        "conj_1class": {
            "mapping_dict": {
                False: 0,
                True: 1
            },
            "rdkit_func": Bond.GetIsConjugated
        },
        "ring_1class": {
            "mapping_dict": {
                False: 0,
                True: 1
            },
            "rdkit_func": Bond.IsInRing
        },
        "stereo": {
            "mapping_dict": {
                BondStereo.STEREONONE: 0,
                BondStereo.STEREOANY: 1,
                BondStereo.STEREOZ: 2,
                BondStereo.STEREOE: 3,
                BondStereo.STEREOCIS: 4,
                BondStereo.STEREOTRANS: 5,
            },
            "rdkit_func": Bond.GetStereo
        }
    }
}

special_feature_convert_yaml = {
    HybridizationType.UNSPECIFIED: 'HybridizationType.UNSPECIFIED',
    HybridizationType.S: 'HybridizationType.S',
    HybridizationType.SP: 'HybridizationType.SP',
    HybridizationType.SP2: 'HybridizationType.SP2',
    HybridizationType.SP3: 'HybridizationType.SP3',
    HybridizationType.SP3D: 'HybridizationType.SP3D',
    HybridizationType.SP3D2: 'HybridizationType.SP3D2',
    HybridizationType.OTHER: 'HybridizationType.OTHER',
    ChiralType.CHI_UNSPECIFIED: 'ChiralType.CHI_UNSPECIFIED',
    ChiralType.CHI_TETRAHEDRAL_CW: 'ChiralType.CHI_TETRAHEDRAL_CW',
    ChiralType.CHI_TETRAHEDRAL_CCW: 'ChiralType.CHI_TETRAHEDRAL_CCW',
    ChiralType.CHI_OTHER: 'ChiralType.CHI_OTHER',
    BondType.SINGLE: 'BondType.SINGLE', 
    BondType.DOUBLE: 'BondType.DOUBLE', 
    BondType.TRIPLE: 'BondType.TRIPLE', 
    BondType.AROMATIC: 'BondType.AROMATIC',
    BondStereo.STEREONONE: 'BondStereo.STEREONONE',
    BondStereo.STEREOANY: 'BondStereo.STEREOANY',
    BondStereo.STEREOZ: 'BondStereo.STEREOZ',
    BondStereo.STEREOE: 'BondStereo.STEREOE',
    BondStereo.STEREOCIS: 'BondStereo.STEREOCIS',
    BondStereo.STEREOTRANS: 'BondStereo.STEREOTRANS',
}


class SingleMolFeature:
    def __init__(self, name, rdkit_func, mapping_dict=None):
        self.name = name
        self.rdkit_func = rdkit_func
        self.mapping = mapping_dict
        self.mapping_one_hot = {}
        if self.mapping is None:
            self.mapping = {}
        if len(self.mapping) != 0:
            self._update_one_hot_mapping()

    def _update_one_hot_mapping(self):
        one_hot_dict = {}
        unique_values = torch.unique(torch.tensor(self.get_values()))
        num_unique_values = unique_values.shape[0]
        for key, value in self.mapping.items():
            one_hot_vec = torch.zeros(num_unique_values)
            one_hot_vec[unique_values==value] = 1
            if not (one_hot_vec.sum().item() == 1):
                raise ValueError("One hot vector sum is not 1. Something went wrong.")
            one_hot_dict[key] = one_hot_vec
        self.mapping_one_hot = one_hot_dict 

    def add_single_mapping(self, key, value):
        if key in self.mapping.keys():
            raise ValueError(f"{key} already in feature mapping of {self.name}.")
        if value in self.mapping.values():
            raise ValueError(f"{value} already included as target value in feature mapping of {self.name}.")
        self.mapping[key] = value
        self._update_one_hot_mapping()

    def map(self, key, one_hot=False):
        if key not in self.mapping.keys():
            raise ValueError(f"Value {key} is not included in feature {self.name} yet. Consider adding it first.")
        if one_hot:
            return self.mapping_one_hot[key]
        return self.mapping[key]

    def get_keys(self):
        return [k for k in self.mapping.keys()]
    
    def get_values(self):
        return [v for v in self.mapping.values()]

    def get_rdkit_func(self):
        return self.rdkit_func

    def get_dim(self):
        return len(self.get_keys())

    def calc_feature(self, obj):
        return self.rdkit_func(obj)

    def calc_one_hot(self, obj):
        feature_key = self.calc_feature(obj)
        return self.map(key=feature_key, one_hot=True)

    def get_mapping(self):
        return self.mapping

    def get_name(self):
        return self.name

    def set_rdkit_func(self, rdkit_func):
        self.rdkit_func = rdkit_func

    def set_mapping(self, mapping):
        self.mapping = mapping

    def set_name(self, name):
        self.name = name


class MolFeatures:
    def __init__ (self):
        self.features = {}

    def __call__(self, name):
        return self.features[name]
    
    def to_dict(self):
        mol_feature_mapping_dict = {}
        for feat_name, feat in self.features.items():
            mol_feature_mapping_dict[feat_name] = feat.get_mapping()
        return mol_feature_mapping_dict
    
    def from_dict(self, feat_type, mol_feature_mapping_dict):
        if not (self.features == {}):
            print(f"Warning: Mol features object is not empty. Features are overwritten by loading from dict.")
            self.features = {}
        for feat_name, mapping in mol_feature_mapping_dict.items():
            self.add_feature(
                name=feat_name,
                rdkit_func=feature_dict[feat_type][feat_name]["rdkit_func"],
                mapping_dict=mapping
            )

    def to_yaml(self):
        mol_feature_mapping_dict = self.to_dict()
        mol_feature_mapping_yaml = copy.deepcopy(mol_feature_mapping_dict)
        for feat_name, mapping in mol_feature_mapping_dict.items():
            for feat_key in mapping.keys():
                # check if object is compatible with yaml serialization
                # if not: translation dict is required, TODO: check alternative of adding individual tags to yaml
                if not (type(feat_key) in [float, int, bool, str, map, type(None)]):
                    yaml_feat_key = special_feature_convert_yaml[feat_key]
                    mol_feature_mapping_yaml[feat_name][yaml_feat_key] = mol_feature_mapping_yaml[feat_name].pop(feat_key)
        return mol_feature_mapping_yaml

    def from_yaml(self, feat_type, mol_feature_mapping_yaml):
        invert_special_feature_convert_yaml = {v: k for k, v in special_feature_convert_yaml.items()}
        mol_feature_mapping_dict = copy.deepcopy(mol_feature_mapping_yaml)
        for feat_name, mapping in mol_feature_mapping_yaml.items():
            for yaml_feat_key in mapping.keys(): 
                # check if object is in translation dict (see to_yaml())
                if yaml_feat_key in invert_special_feature_convert_yaml.keys():
                    feat_key = invert_special_feature_convert_yaml[yaml_feat_key]
                    mol_feature_mapping_dict[feat_name][feat_key] = mol_feature_mapping_dict[feat_name].pop(yaml_feat_key)
        self.from_dict(feat_type=feat_type, mol_feature_mapping_dict=mol_feature_mapping_dict)
        
    def add_feature(self, name, rdkit_func, mapping_dict=None):
        if name in self.features.keys():
            raise ValueError(f"{name} already in feature mapping.")
        new_feature =  SingleMolFeature(name=name,  rdkit_func=rdkit_func, mapping_dict=mapping_dict)
        self.features[name] = new_feature
    
    def add_feature_value(self, name, key, value, rdkit_func=None):
        if name not in self.features.keys():
            if rdkit_func is None:
                raise ValueError(f"{name} feature is not included in features yet but no `rdkit_func` provided.")
            print(f"Warning: {name} feature is not included in features yet. It is added now.")
            new_feature = SingleMolFeature(name, {}, rdkit_func)
            self.features[name] = new_feature
        elif rdkit_func is not None:
            raise ValueError(f"{name} feature already existing in features but arg `rdkit_func` is not None.")
        self.features[name].add_value(key, value)

    def calc_feature_one_hot_vec(self, obj):
        feature_vec = torch.tensor([])
        for f in self.get_all_features():
            feat_one_hot_vec = f.calc_one_hot(obj)
            feature_vec = torch.cat([feature_vec, feat_one_hot_vec])
        return feature_vec

    def get_feature(self, name):
        return self.features[name]

    def get_feature_names(self):
        return [k for k in self.features.keys()]

    def get_all_features(self):
        return [v for v in self.features.values()]

    def get_total_dim(self):
        return sum([f.get_dim() for f in self.get_all_features()])
    
    # TODO: do we need setter methods? for now no reason to change the mapping or values


class AtomFeatures(MolFeatures):
    def __init__(self):
        super().__init__()
        for feat_name, feat_items in feature_dict["atom"].items():
            self.add_feature(
                name=feat_name,
                **feat_items,
            )

class BondFeatures(MolFeatures):
    def __init__(self):
        super().__init__()
        for feat_name, feat_items in feature_dict["bond"].items():
            self.add_feature(
                name=feat_name,
                **feat_items,
            )
