#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '0')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import re,sys,os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Set font to Arial
matplotlib.rcParams['font.family'] = 'Arial'

# Optional: Set PDF to embed fonts (for Illustrator editing)
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better editing


# In[ ]:


from omicsone_streamlit.utils.diff import compare_two_groups
from omicsone_streamlit.plots.volcano import plot_volcano


# In[ ]:


DATA_DIR = Path(r"E:\lab\cptac-prostate\runs\20260401_quality_control")
OUTPUT_DIR = Path(r"E:\lab\cptac-prostate\runs\20260401_mspycloud_cptac_protein_tmt_pca")
OUTPUT_DIR.mkdir(exist_ok=True)
print(DATA_DIR.exists(), OUTPUT_DIR.exists())
meta_path = DATA_DIR / "20_MetaData_03_12_2026_renamed.csv"


# In[ ]:


meta = pd.read_csv(meta_path)
meta_normal = meta[meta["Tissuetype"] == "normal"]
meta_tumor = meta[meta["Tissuetype"] == "tumor"]
purity_map = dict(zip(meta["SampleID"], meta["FirstCategory"]))
stage_map = dict(zip(meta["SampleID"], meta["stage"]))
gleason_map = dict(zip(meta["SampleID"], meta["Grade_Group"]))
# tumors_included = meta_tumor['SampleID'].tolist()
normals = meta_normal['SampleID'].tolist()


# In[ ]:


meta


# In[ ]:


tumors_included = meta[(meta['Tissuetype'] == 'tumor')&(meta['FirstCategory'] == "Sufficient Purity")]['SampleID'].tolist()


# In[ ]:


# data
files = [i for i in DATA_DIR.iterdir() if i.is_file()]
files = [i for i in files if re.search(r"mspycloud_Global_DDA_TMT_log2ratio",i.stem, re.IGNORECASE) \
    and not re.search(r"tumor_purity_corrected",i.stem, re.IGNORECASE)]
data_path = files[0]
protein_header_cols = ["Protein.Group.Accessions"]
data = pd.read_csv(data_path,sep="\t").set_index(protein_header_cols)
data_nomiss = data.dropna()


# In[ ]:


data


# In[ ]:


total_genes = sorted(set([i.split("|")[-1].split(" ")[0] for i in data.index]))
print(len(total_genes))


# In[ ]:


print(len(tumors_included), len(normals))


# In[ ]:


fc = 1.5
diff = compare_two_groups(data, tumors_included, normals, 
                                method="Wilcoxon(Unpaired)",
                                max_miss_ratio_global=0.5, 
                                max_miss_ratio_group=0.5,
                                fdr_cutoff=0.01, 
                                log2fc_cutoff=np.log2(fc))


# In[ ]:


diff


# In[ ]:


diff['Significance'] = diff['Significance'].apply(lambda x: 'NS' if pd.isna(x) else x)


# In[ ]:


xmin = diff['Log2FC(median)'].min()
xmax = diff['Log2FC(median)'].max()
print(xmin, xmax)
xlimit = np.max([abs(np.floor(xmin)), np.ceil(xmax)])
print(xlimit)


# In[ ]:


fig = plot_volcano(diff,
                   log2fc_threshold=np.log2(fc),
                   xlim = (-1 * xlimit, xlimit))


# In[ ]:


diff[diff['Significance'] == 'S-U']


# In[ ]:


diff2 = diff.copy()
diff2.index = [ i.split("|")[-1].split(" ")[0] for i in diff.index ]


# In[ ]:


from omicsone_streamlit.utils.pathway import omicsone_enrichr


# In[ ]:


su_genes = diff2[diff2['Significance'] == 'S-U'].index.tolist()
sd_genes = diff2[diff2['Significance'] == 'S-D'].index.tolist()
u_genes = diff2[diff2['Significance'] == 'U'].index.tolist()
d_genes = diff2[diff2['Significance'] == 'D'].index.tolist()
print(len(su_genes), len(sd_genes), len(u_genes), len(d_genes))


# In[ ]:


out_dir = OUTPUT_DIR / "genes_protein_tumor_vs_normal"
out_dir.mkdir(exist_ok=True)
with open(out_dir / "S-U_genes.txt", "w") as f:
    f.write("\n".join(su_genes))
    f.close()
with open(out_dir / "S-D_genes.txt", "w") as f:
    f.write("\n".join(sd_genes))
    f.close()
with open(out_dir / "U_genes.txt", "w") as f:
    f.write("\n".join(u_genes))
    f.close()
with open(out_dir / "D_genes.txt", "w") as f:
    f.write("\n".join(d_genes))
    f.close()


# In[ ]:


import gseapy
names = gseapy.get_library_name()
[i for i in names if "Hallmark" in i]


# In[ ]:


out_dir = OUTPUT_DIR / "enrichr_protein_tumor_vs_normal_hallmark"
out_dir.mkdir(exist_ok=True)
omicsone_enrichr(su_genes, total_genes=total_genes, 
                 gene_sets="MSigDB_Hallmark_2020", 
                 fdr = 0.05,
                 outdir=out_dir)


# In[ ]:


fig = plot_volcano(diff2,
                   log2fc_threshold=np.log2(fc),
                   xlim = (-1 * xlimit, xlimit),
                   annotations=annotatins,
                   data_type ='protein')


# In[ ]:




