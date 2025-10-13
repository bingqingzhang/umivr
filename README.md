# Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization

> Despite recent advances, Text-to-video retrieval (TVR) is still hindered by multiple inherent uncertainties, such as ambiguous textual queries, indistinct text-video mappings, and low-quality video frames. Although interactive systems have emerged to address these challenges by refining user intent through clarifying questions, current methods typically rely on heuristic or ad-hoc strategies without explicitly quantifying these uncertainties, limiting their effectiveness. Motivated by this gap, we propose UMIVR, an Uncertainty-Minimizing Interactive Text-to-Video Retrieval framework that explicitly quantifies three critical uncertainties-text ambiguity, mapping uncertainty, and frame uncertainty-via principled, training-free metrics: semantic entropy-based Text Ambiguity Score (TAS), Jensen-Shannon divergence-based Mapping Uncertainty Score (MUS), and a Temporal Quality-based Frame Sampler (TQFS). By adaptively generating targeted clarifying questions guided by these uncertainty measures, UMIVR iteratively refines user queries, significantly reducing retrieval ambiguity. Extensive experiments on multiple benchmarks validate UMIVR's effectiveness, achieving notable gains in Recall@1 (69.2% after 10 interactive rounds) on the MSR-VTT-1k dataset, thereby establishing an uncertainty-minimizing foundation for interactive TVR.

---

## 📚 Data Preparation

1.  **Download Videos**: Please download the required dataset videos (e.g., MSR-VTT, DiDeMo) by following the official instructions provided in the [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip?tab=readme-ov-file#data-preparing).

2.  **Annotation Files**: We have conveniently included all necessary annotation files in the `anno/` directory of this repository.

## 🔧 Installation

### 1. Prerequisites
- Python >= 3.10
- PyTorch == 2.0.1
- CUDA >= 11.7

### 2. Setup Instructions
First, clone the repository and navigate into the project directory:
```bash
git clone https://github.com/bingqingzhang/umivr.git
cd umivr
````

Next, create and activate a Conda environment:

```bash
conda create -n umivr python=3.10 -y
conda activate umivr
```

Finally, install the required dependencies. We recommend installing them in the following order:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install decord opencv-python==4.9.0.80
pip install git+https://github.com/facebookresearch/pytorchvideo

pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.2/flash_attn-2.5.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

pip install -e .
```

## 🚀 Running Inference

> Before running the code, you must specify the path to your downloaded datasets. Open the relevant config file in the `retrieval_config/` directory (e.g., `umivr_msrvtt.json`) and update the `video_path` field to point to the root directory of your dataset videos.

To run interactive retrieval with **UMIVR** on the MSR-VTT-1kA dataset, use the following command:

```
python interact_ivr.py --config-file retrieval_config/umivr_msrvtt.json --absolute-file-path 'path/to/root_directory_of_dataset'
```

To run other baseline methods, simply change the `--config-file` argument to use the corresponding JSON file from the `retrieval_config/` directory.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zhang2025umivr,
 title={Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization},
 author={Zhang, Bingqing and Cao, Zhuo and Du, Heming and Li, Yang and Li, Xue and Liu, Jiajun and Wang, Sen},
 booktitle={ICCV2025},
 year={2025}
}
```

## 🙏 Acknowledgement

Our codebase is built upon the excellent work from the following repositories. We extend our gratitude to their authors.

  - [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
  - [IVR-QA-Baselines](https://github.com/kevinliang888/IVR-QA-baselines)