# LarSPI

As training a good feature extraction model requires a large variety of data, we prepared a large-scale unlabeled dataset(LarSPI) by reorganizing the whole slide images in TCGA. We kept only
 the whole slide images (WSI) at 40× magnification. This is because the size of the nuclei matters a lot in classification, and the mixing of magnifications may corrupt the classification model. We
  randomly cropped a foreground patch of size 1024 × 1024 from each selected WSIs and manually reviewed them to remove those of very poor quality, e.g., those with severe stains. We
   then used the trained [HoVer-Net](https://github.com/vqdang/hover_net) to detect and segment the nucleus instances in the remaining patches. Finally, 20,187 high-quality patches of size 1024 × 1024, each from a different WSI
   , accompanied by its nucleus mask, make up our dataset LarSPI. Note that we also preserve the cancer types and slide types in the filenames, following the TCGA dataset. The size of our
    dataset is very large compared to other existing datasets such as [PanNuke](https://jgamper.github.io/PanNukeDataset/) or [CoNSeP](https://github.com/vqdang/hover_net). Details of our dataset and a comparison between our dataset
     and these two datasets are shown
     in Table below.
 
 |           |Cancer Types|Patch Size  |  No. Patches|No. Classes| 
| -----------|----------- | -----------|-----------|-----------|
| [CoNSeP](https://github.com/vqdang/hover_net)       | 1      | 1000x1000 | 41      |7     |
| [PanNuke](https://jgamper.github.io/PanNukeDataset/)| 19     | 256x256   | 7901    |5     |
| LarSPI                                              | 32     | 1024x1024 | 20187   |n/a   |

## Download Link
We provide two download links here for the LarSPI dataset.

[Onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/wenhua00_connect_hku_hk/Es00Ck815j1CgUEq8joixWABUivMT2wvC-8gnt_fH2K4Aw?e=GiDF1F)

[BaiduDisk](https://pan.baidu.com/s/1WBUFxA6nsiEB6VwWpUj6OA?pwd=111n)

## Repository Structure

Below are the main directories in the repository: 

- `imgs/`: the folder contains pathology patches from different types of diseases at the resolution of 40x
- `mask/`: the folder contains segmentation masks for the patches of the *imgs* folder
- `overlay/`: the folder contains visualization overlay for the segmentation masks on the patches
- `draw_overlay.py`: the python scripts used to generate the overlay images

## Usage and Options
### Mask format
The masks in the *mask* folder are numpy arrays of size 1024x1024. Each mask is a pixel-wise correspondense to the image patch of the same name in the *imgs* folder. We also include the
 *draw_overlay.py* as an example to show how to use this data.
 
### Cancer Types
The cancer types are indicated by the folder names in the *imgs* folder and the *mask* folder. They are abbreviation for 32 different types of diseases. 

|Abbreviation|Cancer Type| Abbreviation|Cancer Type|
| -----------|----------- | -----------|-----------|
|acc_tcga    |Adrenocortical carcinoma    |lusc_tcga |Lung squamous cell carcinoma|
|blca_tcga   |Bladder Urothelial Carcinoma|meso_tcga |Mesothelioma|
|brca_tcga   |Breast invasive carcinoma   |ov_tcga   |Ovarian serous cystadenocarcinoma|
|cesc_tcga   |Cervical squamous cell <br/>carcinoma and endocervical <br/> adenocarcinoma|paad_tcga|Pancreatic adenocarcinoma|
|chol_tcga   |Cholangiocarcinoma          |pcpg_tcga |Pheochromocytoma and Paraganglioma|
|coad_tcga   |Colon adenocarcinoma        |prad_tcga |Prostate adenocarcinoma|
|dlbc_tcga   |Lymphoid Neoplasm Diffuse Large B-cell Lymphoma|read_tcga|Rectum adenocarcinoma|
|esca_tcga   |Esophageal carcinoma        |sarc_tcga |Sarcoma|
|gbm_tcga    |Glioblastoma multiforme     |skcm_tcga |Skin Cutaneous Melanoma|
|hnsc_tcga   |Head and Neck squamous cell carcinoma|stad_tcga|Stomach adenocarcinoma|
|kich_tcga   |Kidney Chromophobe          |tgct_tcga |Testicular Germ Cell Tumors|
|kirc_tcga   |Kidney renal clear cell carcinoma|thca_tcga|Thyroid carcinoma|
|kirp_tcga   |Kidney renal papillary cell carcinoma|thym_tcga|Thymoma|
|lgg_tcga    |Brain Lower Grade Glioma    |ucec_tcga |Uterine Corpus Endometrial Carcinoma|
|lihc_tcga   |Liver hepatocellular carcinoma|ucs_tcga |Uterine Carcinosarcoma|
|luad_tcga   |Lung adenocarcinoma|uvm_tcga |Uveal Melanoma|

### Slide Types
The image filenames contains the information of whether the slide is frozen or Formalin-Fixed Paraffin-Embedded.
For example

TCGA-OR-A5J1-01A-01-**TS**1.CFE08710-54B8-45B0-86AE-500D6E36D8A5_8192_76800_0.248.png

shows that this patch is from a frozen slide. The table for the slide types is as below.

|Abbreviation|Slide Type| Abbreviation|Slide Type|
| -----------|--------- | -----------|-----------|
| TS         |Frozen    | DX         | FFPE      |
| BS         |Frozen    | MS         | Frozen    |


## License

The dataset provided here is for research purposes only. Commercial use is not allowed. The data is held under the following license:
[Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

```
@ARTICLE{9869632,
  author={Zhang, Wenhua and Zhang, Jun and Yang, Sen and Wang, Xiyue and Yang, Wei and Huang, Junzhou and Wang, Wenping and Han, Xiao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Knowledge-Based Representation Learning for Nucleus Instance Classification from Histopathological Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3201981}}
```

## Authors

* [Wenhua Zhang](https://github.com/WinnieLaugh)
