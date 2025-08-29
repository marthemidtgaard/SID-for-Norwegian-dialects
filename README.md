# Slot and Intent Detection (SID) for Norwegian Dialects

This repository contains code used for the thesis titled [Slot and Intent Detection for Norwegian Dialects](https://www.duo.uio.no/handle/10852/120951). The codebase is aslo used for our participation in the VarDial 2025 Shared Task on Norwegian Slot and Intent Detection, named [LTG at VarDial 2025 NorSID: More and Better Training Data for Slot and Intent Detection](https://aclanthology.org/2025.vardial-1.15/) (Midtgaard et al., 2025). Some scripts, settings and datasets have been further improved and extended as part of the master's thesis work, compared to the original shared task submission.

## 📂 Structure
- `master_thesis/`: Training scripts, configuration files, data processing scripts, and all datasets. 
- `machamp/`: Toolkit used for fine-tuning the SID models.
- `simalign/`: Word alignment tool used for re-aligning data.
- `vardial_data/`: New and improved Norwegian SID annotated dataset introduced and used in the VarDial shared task submission.

## 🛠️ Code
Clone with the requrse-submodules flag to download all submodule contents: `git clone --recurse-submodules https://github.com/marthemidtgaard/SID-for-Norwegian-dialects.git`. The code and data to run all experiments is located inside the `master_thesis/` folder. All experiments are run on Fox Educloud, using module load commands inside SLURM job scripts to manage dependencies. A `requirements.txt` is provided for running the code outside Educloud.

## ✍️ Citation
If you use this code, please cite:
- Midtgaard et al., 2025. [LTG at VarDial 2025 NorSID: More and Better Training Data for Slot and Intent Detection](https://aclanthology.org/2025.vardial-1.15/)
- Midtgaard, 2025. [Slot and Intent Detection for Standard and Dialectal Norwegian](https://www.duo.uio.no/handle/10852/120951)

