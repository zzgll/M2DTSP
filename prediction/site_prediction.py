# %%
import os
import numpy as np
import torch
from torch_geometric.data import Batch
import pandas as pd
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG
import cairosvg
import cv2
import matplotlib.cm as cm
from tqdm import tqdm

from model import M2DTSP
from dataset import *
from utils import *
from Bio.PDB import PDBParser

class GradAAM():
    def __init__(self, model, molecule_module, protein_module):
        self.model = model
        self.molecule_module = molecule_module
        self.protein_module = protein_module
        self.target_feat = None
        self.protein_feat = None

        # 为药物分子模块注册hook
        molecule_module.register_forward_hook(self.save_molecule_hook)
        # 为蛋白质模块注册hook
        protein_module.register_forward_hook(self.save_protein_hook)

    def save_molecule_hook(self, molecule_module, input, output):
        # 保存药物分子的特征
        self.target_feat = output.x
        print(output.protein)

    def save_protein_hook(self, module, input, output):
        # 保存蛋白质的特征
        self.protein_feat = output.x
        print(self.protein_feat)

    def __call__(self, data):
        self.model.eval()
        output = self.model(data).view(-1)
        # 设置retain_graph=True以允许二次反向传播
        with torch.enable_grad():  # 确保在计算梯度时梯度被激活
            if self.target_feat.requires_grad == False:
                self.target_feat.requires_grad_(True)



        # 计算药物分子的梯度加权
        if self.target_feat is not None:
            grad = torch.autograd.grad(output, self.target_feat, retain_graph=True)[0]
            channel_weight = torch.mean(grad, dim=0, keepdim=True)
            channel_weight = normalize(channel_weight)
            weighted_feat = self.target_feat * channel_weight
            cam_drug = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
            cam_drug = normalize(cam_drug)
        else:
            cam_drug = None

        # 计算蛋白质节点的权重
        if self.protein_feat is not None:
            protein_grad = torch.autograd.grad(output, self.protein_feat, retain_graph=True)[0]
            protein_channel_weight = torch.mean(protein_grad, dim=0, keepdim=True)
            protein_channel_weight = normalize(protein_channel_weight)
            protein_weighted_feat = self.protein_feat * protein_channel_weight
            protein_weights = torch.sum(protein_weighted_feat, dim=-1).detach().cpu().numpy()
            protein_weights = normalize(protein_weights)
        else:
            protein_weights = None

        return output.detach().cpu().numpy(), cam_drug, protein_weights

def clourMol(mol,highlightAtoms_p=None,highlightAtomColors_p=None,highlightBonds_p=None,highlightBondColors_p=None,sz=[400,400], radii=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p,highlightAtomColors=highlightAtomColors_p, highlightBonds= highlightBonds_p,highlightBondColors=highlightBondColors_p, highlightAtomRadii=radii)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    res = cairosvg.svg2png(svg.data, dpi = 600, output_width=2400, output_height=2400)
    nparr = np.frombuffer(res, dtype=np.uint8)
    segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return segment_data

def main():
    device = torch.device('cuda:0')

    fpath = os.path.join('data', 'full_toxcast')
    test_df = pd.read_csv(os.path.join(fpath, 'raw', 'full_pdbv3.csv'))
    test_set = GNNDataset(fpath, train=False)

    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    try:
        load_model_dict(model, 'pretrained_model/epoch-258, loss-0.0678, cindex-0.9539, val_loss-0.2347, test_loss-0.2328.pt')
    except:
        model_dict = torch.load('pretrained_model/epoch-258, loss-0.0678, cindex-0.9539, val_loss-0.2347, test_loss-0.2328.pt')
        for key, val in model_dict.copy().items():
            if 'lin_l' in key:
                new_key = key.replace('lin_l', 'lin_rel')
                model_dict[new_key] = model_dict.pop(key)
            elif 'lin_r' in key:
                new_key = key.replace('lin_r', 'lin_root')
                model_dict[new_key] = model_dict.pop(key)
        model.load_state_dict(model_dict)

    gradaam = GradAAM(model, molecule_module=model.ligand_encoder.features.transition3, protein_module = model.protein_encoder.features.transition3)


    bottom = cm.get_cmap('Blues_r', 256)
    top = cm.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    smile_list = list(test_df['compound_iso_smiles'].unique())

    progress_bar = tqdm(total=len(smile_list))

    for idx in range(len(test_set)):
        smile = test_df.iloc[idx]['compound_iso_smiles']

        if len(smile_list) == 0:
            break
        if smile in smile_list:
            smile_list.remove(smile)
        else:
            continue

        data = Batch.from_data_list([test_set[idx]])
        #print(type(test_set[idx]))
        data = data.to(device)
        _, atom_att, protein_att = gradaam(data)

        mol = Chem.MolFromSmiles(smile)
        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        img = clourMol(mol,highlightAtoms_p=range(len(atom_att)), highlightAtomColors_p=atom_color, radii=radii)

        cv2.imwrite(os.path.join('results', f'{idx}.png'), img)
        # 下面是蛋白质可视化
        B_factor = protein_att * 100

        protein = Batch.from_data_list(data.protein)

        protein = protein.protein_name

        protein = protein[0]

        # 读取PDB文件
        # pdb_file = '/home/best/DataDTA/MGraphDTA/filtered_davis/pdb_new/' + protein + '.pdb'
        pdb_file = protein + '.pdb'
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        with open(pdb_file, 'r') as f:
            pdb_lines = f.readlines()

        start_residue_num = -1
        atom_to_color = dict()
        residue_in_set = set()
        # 打开一个新的PDB文件用于写入
        for i, line in enumerate(pdb_lines):
            if line.startswith('ATOM'):
                residue_name = line[25:26]
                if residue_name not in residue_in_set:
                    residue_in_set.clear()
                    residue_in_set.add(residue_name)
                    start_residue_num += 1
                if start_residue_num >= len(B_factor):
                    continue
                atom_to_color[i] = B_factor[start_residue_num]

                # print(residue_name)
        with open('colored_' + protein + '.pdb', 'w') as f:
            for i, line in enumerate(pdb_lines):
                # 在ATOM记录行中添加颜色信息到B-factor字段
                if line.startswith('ATOM'):
                    # 将颜色信息转换为B-factor值
                    bfactor_value = atom_to_color[i]  # 使用适当的数据
                    # 格式化B-factor字段，通常限定到6位小数
                    bfactor_field = f'{bfactor_value:.2f}'
                    # 替换B-factor字段
                    # line = line[:60] + bfactor_field + line[66:]
                    line = line[:60] + ' ' + bfactor_field + line[66:]
                # 写入新的行到新的PDB文件
                f.write(line)

    progress_bar.update(1)



if __name__ == '__main__':
    main()

