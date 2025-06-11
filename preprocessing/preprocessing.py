import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
import pickle


VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target] 

class GNNDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self, data_path):
        # 首先是读取数据
        df = pd.read_csv(data_path)
        # smiles数据先读取到
        smiles = df['compound_iso_smiles'].unique()
        sequence_proteins = df['target_sequence'].unique()
        # 图是一个字典
        graph_dict = dict()

        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            # smiles数据先转换成mol数据
            g = self.mol2graph(mol)
            # 每一个smiles字符串对应一个图数据
            graph_dict[smile] = g
        protein_dict = dict()
        for protein in sequence_proteins:

            g = self.protein2graph('1a0q')
            protein_dict[protein] = g

        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['affinity']

            if graph_dict.get(smi) == None:
                print(smi)
                delete_list.append(i)
                continue

            x, edge_index, edge_attr = graph_dict[smi]

            # modify
            x = (x - x.min()) / (x.max() - x.min())

            x_p,edge_index_p,edge_attr_p = protein_dict[sequence]

            x_p = (x_p - x_p.min()) / (x_p.max() - x_p.min())

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len- len(target)))
            else:
                target = target[:target_len]

            # Get Labels
            try:
                protein_data = DATA.Data(
                    x=x_p,
                    edge_index=edge_index_p,
                    edge_attr=edge_attr_p,
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target])
                )

                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.FloatTensor([label]),
                    target=protein_data
                )

            except:
                    print("unable to process: ", smi)

            data_list.append(data)

        df = df.drop(delete_list, axis=0, inplace=False)
        df.to_csv(data_path, index=False)

        return data_list

    def process(self):
        data_list = self.process_data(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {} # 用于存储边的属性信息
        # 边形成一个onehot索引，包含了是否是化学键 SINGLE，DOUBLE，TRIPLE，AROMATIC，非共轭IsConjugated，共轭IsConjugated
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1) #这个是边的index
        edge_attr = torch.FloatTensor(list(e.values())) # 这个是边的属性
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        # 通过rdkit获取mol里面的特征数据
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        # 图是nx里面的DiGraph(),DiGraph是有向图。有向图是一种图结构，其中边具有方向性，从一个节点指向另一个节点。
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)

            # 在图上增加节点：节点特征包括。节点符号，原子数目
            # print("atom_i.GetAtomicNum()",atom_i.GetAtomicNum())
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        # 供体的节点全部改成1， 受体的相邻节点全部改成1
        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                # 如果ij之间存在化学键
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    # b_type存储化学键类型，IsConjugated表示化学键是否共轭
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)
        return node_attr, edge_index, edge_attr

    def protein2graph(self, target):
        file_name = target + "_protein.pdb"
        dssp_path = target + ".dssp"
        print(target)
        feat = []
        parser = PDBParser()
        structure = parser.get_structure(target, file_name)
        model = structure[0]
        dssp = DSSP(model, dssp_path)
        sr = ShrakeRupley()
        sr.compute(structure, level="R")
        chains = structure.get_chains()
        feature_vector = []

        g = nx.DiGraph()
        chain_dict = dict()
        for chain in chains:  # 根据实际情况选择链
            residues = chain.get_residues()
            chain_id = chain.get_id()
            print("chain_id : ", chain_id)
            one_chain_list = []
            for residue in residues:
                residue_number = residue.get_id()[1]
                print(residue_number)
                # 获取残基序号
                solventAccessiblesurface = round(residue.sasa, 2)
                print(solventAccessiblesurface)
                one_chain_list.append([solventAccessiblesurface])
            chain_dict[chain_id] = one_chain_list
        print(chain_dict)
        chain_dict_for_psiandphi = dict()
        chains = structure.get_chains()
        for chain in chains:  # 根据实际情况选择链
            chain_id = chain.get_id()
            print("chain_id : ", chain_id)
            chain_list = residue_from_dssp(dssp, chain_id)
            chain_dict_for_psiandphi[chain_id] = chain_list

        chain_dict_for_one_hot_secondary_structure = dict()
        chains = structure.get_chains()
        for chain in chains:  # 根据实际情况选择链
            chain_id = chain.get_id()
            print("chain_id : ", chain_id)
            chain_list = onehot_structure_from_dssp(dssp, chain_id)
            chain_dict_for_one_hot_secondary_structure[chain_id] = chain_list
            # pass
        print(chain_dict_for_one_hot_secondary_structure)

        chain_dict_for_aaphy7_and_blosum62 = dict()
        chains = structure.get_chains()
        for chain in chains:  # 根据实际情况选择链
            chain_id = chain.get_id()
            print("chain_id : ", chain_id)
            chain_list = aaphy7_and_blosum62_from_dssp(dssp, chain_id)
            chain_dict_for_aaphy7_and_blosum62[chain_id] = chain_list
            # pass
        print(chain_dict_for_aaphy7_and_blosum62)

        merged_dict = dict()
        for key in chain_dict:
            value1 = chain_dict[key]
            value2 = chain_dict_for_psiandphi[key]
            value3 = chain_dict_for_one_hot_secondary_structure[key]
            value4 = chain_dict_for_aaphy7_and_blosum62[key]
            # value5 = chain_dict_for_acceptor_and_donor[key]
            merged_list = []
            for item1, item2, item3, item4 in zip(value1, value2, value3, value4):
                merged_list_list = item1 + item2 + item3 + item4
                merged_list.append(merged_list_list)
            # merged_value = np.concatenate(value1,value2)
            # merged_value = [value1[i] + value2[i] for i in range(len(value1))]
            merged_dict[key] = merged_list

        print(merged_dict)
        from proteingraph import read_pdb

        graph = read_pdb(file_name)
        # 遍历PDB结构中的链
        edge_attr,node_str_list = construct_edge_attr(graph)

        node_dict = change_node_attr_structure(merged_dict,structure,node_str_list)
        str_2_int_node_dict = dict()
        node_num = 0
        for node in node_dict:
            str_2_int_node_dict[node] = node_num
            node_num += 1
        node_num = len(node_dict)

        print(chain_dict_for_psiandphi)

        node_list = np.zeros((node_num,39))
        for node in node_dict:
            node_list[str_2_int_node_dict[node]] = np.array(node_dict[node])

        edge_list = np.zeros((len(edge_attr),2))
        edge_attr_list = np.zeros((len(edge_attr),6))
        tuple_2_num_dict = dict()
        num_2_ndarray_dict = dict()

        num_2_edge_attr = dict()
        edge_num = 0
        for edge in edge_attr:
            node_2_edge = np.zeros(2)
            tuple_2_num_dict[edge] = edge_num
            node_2_edge[0] = str_2_int_node_dict[edge[0]]
            node_2_edge[1] = str_2_int_node_dict[edge[1]]
            num_2_edge_attr[edge_num] = np.array(edge_attr[edge])
            num_2_ndarray_dict[edge_num] = node_2_edge
            edge_num += 1
        edge_len = edge_num
        for i in range(edge_len):
            edge_list[i] = num_2_ndarray_dict[i]
        for i in range(edge_len):
            edge_attr_list[i] = num_2_edge_attr[i]
        node_attr = torch.FloatTensor(node_list)
        edge_index = torch.LongTensor(edge_list).transpose(0, 1)  # 这个是边的index
        edge_attr = torch.FloatTensor(edge_attr_list)  # 这个是边的属性

        return node_attr,edge_index,edge_attr
        # with open("protein.pkl","wb") as file:
        #     pickle.dump(out_dict,file)
        # print(file)

# 定义二级结构编码映射
structure_mapping = {
    'G': [1, 0, 0, 0, 0, 0],  # 无结构
    'H': [0, 1, 0, 0, 0, 0],  # α-螺旋
    'B': [0, 0, 1, 0, 0, 0],  # 孤立β-桥残基
    'E': [0, 0, 1, 0, 0, 0],  # β-链
    'I': [0, 0, 0, 1, 0, 0],  # 3-10 螺旋
    'T': [0, 0, 0, 0, 1, 0],  # 转角
    'S': [0, 0, 0, 0, 0, 1]  # 弯曲
}


def residue_from_dssp(dssp, chain_id):
    chain_list = []
    for residue_keys in dssp.keys():

        residue = dssp[residue_keys]
        chain_id_dssp = residue_keys[0]
        phi_angle = residue[4]
        psi_angle = residue[5]
        if chain_id_dssp == chain_id:
            chain_list.append([phi_angle, psi_angle])
    return chain_list

    # pass


def onehot_structure_from_dssp(dssp, chain_id):
    chain_list = []
    for residue_keys in dssp.keys():
        residue = dssp[residue_keys]
        struct = residue[2]
        chain_id_dssp = residue_keys[0]
        if chain_id_dssp == chain_id:
            if struct == 'G' or struct == 'H' or struct == 'B' or struct == 'E' or struct == 'I' or struct == 'T' or struct == 'S':
                one_hot = structure_mapping[struct]
            else:
                one_hot = structure_mapping['G']
            chain_list.append(one_hot)

    return chain_list


aaphy7_mapping = {
    'A': [-0.35, -0.68, -0.677, -0.171, -0.17, 0.9, -0.476],
    'C': [-0.14, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
    'D': [-0.213, -0.417, -0.281, -0.767, -0.9, -0.155, -0.635],
    'E': [-0.23, -0.241, -0.058, -0.696, -0.868, 0.9, -0.582],
    'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
    'G': [-0.9, -0.9, -0.9, -0.342, -0.179, -0.9, -0.9],
    'H': [0.384, 0.11, 0.138, -0.271, 0.195, -0.031, -0.106],
    'I': [0.9, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
    'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
    'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
    'M': [0.11, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
    'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
    'P': [0.247, -0.9, -0.294, 0.055, -0.01, -0.9, 0.106],
    'Q': [-0.23, -0.11, -0.02, -0.464, -0.276, 0.528, -0.371],
    'R': [0.105, 0.373, 0.466, -0.9, 0.9, 0.528, -0.371],
    'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
    'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
    'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.9],
    'W': [0.479, 0.9, 0.9, 0.9, -0.209, 0.279, 0.529],
    'Y': [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]
}
blosum62_mapping = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1]
}


def aaphy7_and_blosum62_from_dssp(dssp, chain_id):
    chain_list = []
    for residue_keys in dssp.keys():
        residue = dssp[residue_keys]
        amio_acid = residue[1]
        chain_id_dssp = residue_keys[0]
        if chain_id_dssp == chain_id:

            if amio_acid not in aaphy7_mapping:
                aaphy7 = [0.0] * 7
            else:
                aaphy7 = aaphy7_mapping[amio_acid]
            if amio_acid not in blosum62_mapping:
                blosum62 = [0] * 23
            else:
                blosum62 = blosum62_mapping[amio_acid]
            merge_list = aaphy7 + blosum62
            chain_list.append(merge_list)
    return chain_list

acceptor_and_donor_mapping = {
    'A': [1.0, 0.3],
    'C': [1,0, 0.3],
    'D': [1.0, 0.3],
    'E': [1.0, 0.3],
    'F': [0.0, 0.0],
    'G': [0.0, 0.0],
    'H': [1.0, 0.3],
    'I': [0.0, 0.0],
    'K': [1.0, 0.3],
    'L': [0.0, 0.0],
    'M': [1.0, 0.3],
    'N': [1.0, 0.3],
    'P': [0.0, 0.0],
    'Q': [1.0, 0.3],
    'R': [1.0, 0.3],
    'S': [1.0, 0.3],
    'T': [1.0, 0.3],
    'V': [0.0, 0.0],
    'W': [0.0, 0.0],
    'Y': [1.0, 0.3]
}
def acceptor_and_donor_from_map(dssp,chain_id):
    chain_list = []
    for residue_keys in dssp.keys():
        residue = dssp[residue_keys]
        amio_acid = residue[1]
        chain_id_dssp = residue_keys[0]
        if chain_id_dssp == chain_id:
            if amio_acid not in aaphy7_mapping:
                acceptor_and_donor = [0.0] * 2
            else:
                acceptor_and_donor = acceptor_and_donor_mapping[amio_acid]
            chain_list.append(acceptor_and_donor)
    return chain_list
acceptor_and_donor_mapping_V4 = {
    'R': [0, 1, 0, 0],
    'K': [0, 1, 0, 0],
    'W': [0, 1, 0, 0],
    'D': [0, 0, 1, 0],
    'E': [0, 0, 1, 0],
    'N': [1, 0, 0, 0],
    'Q': [1, 0, 0, 0],
    'H': [1, 0, 0, 0],
    'S': [1, 0, 0, 0],
    'T': [1, 0, 0, 0],
    'Y': [1, 0, 0, 0],
}
def acceptor_and_donor_from_map_V4(dssp,chain_id):
    chain_list = []
    for residue_keys in dssp.keys():
        residue = dssp[residue_keys]
        amio_acid = residue[1]
        chain_id_dssp = residue_keys[0]
        if chain_id_dssp == chain_id:
            if amio_acid not in aaphy7_mapping:
                acceptor_and_donor = [0.0] * 3 + [1.0]
            else:
                acceptor_and_donor = acceptor_and_donor_mapping[amio_acid]
            chain_list.append(acceptor_and_donor)
    return chain_list

edge_attr_mapping = {
    'backbone': 0,
    'hydrophobic': 1,
    'ionic' : 2,
    'disulfide': 3,
    'hbond' : 4,
    'aromatic' : 5
}
import numpy as np

def construct_edge_attr(graph):
    print(graph)
    adj_data = graph.adjacency()
    edge_dict = dict()
    list1 = []
    edge_set = set()
    for node, neighbors in adj_data:
        for j in neighbors:
            t = (node,j)
            if t not in edge_set:
                attr_array = np.zeros(6)
                for k in neighbors[j]['kind']:
                    idx = edge_attr_mapping[k]
                    attr_array[idx] = 1
                edge_dict[t] = attr_array
                edge_set.add((j,node))
                list1 += neighbors[j]['kind']
    #     print(node)
    #     print(neighbors)
    # print(list1)
    # set_list = list(set(list1))
    # print(set_list)
    # print(edge_dict)

    node_data = graph.nodes
    node_list = list(node_data)

    return edge_dict,node_list

def change_node_attr_structure(merge_dict,structure,node_str_list):

    node_dict = dict()
    chains = structure.get_chains()
    for chain in chains:
        chain_id = chain.get_id()
        chain_node_list = merge_dict[chain_id]
        if len(chain_node_list) is 0:
            continue
        chain_residue = chain.get_residues()
        edge_start = 2
        node_start = 0
        for residue in chain_residue:
            resname = str(residue.get_resname())
            edge_start = int(residue.get_id()[1])
            node_name = chain_id + str(edge_start) + resname
            node_value = chain_node_list[node_start]
            node_dict[node_name] = node_value
            # edge_start += 1
            node_start += 1
            # if chain_id == 'l'
            print(residue.get_resname())

    return node_dict
def search_edge_num(edge_start,resname,node_str_list,chain_id):
    min_edge_start = (edge_start - 10) if edge_start - 10 >= 0 else 0
    max_edge_start = edge_start + 10
    for i in range(edge_start,max_edge_start+1):
        node_name = chain_id + str(i) + resname
        # print(node_name)
        if node_name in node_str_list:
            return i
    for i in range(edge_start,min_edge_start,-1):
        node_name = chain_id + str(i) + resname
        # print(node_name)
        if node_name in node_str_list:
            return i

    return edge_start


if __name__ == "__main__":
    GNNDataset('data/filtered_davis')
