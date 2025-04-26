In this subfolder you will find our new and improved Norwegian SID datasets used in our (LTG) submission to the VarDial 2025 Shared Task on Slot and Intent Detection (SID). The data includes re-aligned and re-translated Norwegian versions of the [xSID-0.6 dataset](https://github.com/mainlp/xsid), as well as a Norwegian portion of the [MASSIVE dataset](https://github.com/alexa/massive/tree/main) mapped to the xSID annotation scheme. Below is a description of the data files and their contents.

---
### 1. MASSIVE Norwegian Data
- **Description**: The Norwegian subset of the MASSIVE dataset, consisting of manually SID-annotated utterances in standard Bokmål.
- **File**: `nb_mas.json`

### 2. Norwegian xSID Data
#### a) Original Bokmål (`nb`)
- **Description**: The original Norwegian `xSID-0.6` training data. Derived through machine translation and slot alignment via attention mechanisms.
- **File**: `nb.json`
  
#### b) Bokmål Re-Aligned (`nb_ra`)
- **Description**: An automatically re-aligned version of the original Norwegian `xSID-0.6` training dataset. 
- **File**: `nb_ra.json`

#### c) Bokmål Translated and Re-Aligned (`nb_rt`)
- **Description**: A new version of `xSID-0.6` created by first translating English utterances into Bokmål using machine translation and then re-aligning slot spans.
- **File**: `nb_rt.json`

#### d) Nynorsk Translated and Re-Aligned (`nn_rt`)
- **Description**: A new version of `xSID-0.6` created by first translating English utterances into Nynorsk using machine translation and then re-aligning slot spans.
- **File**: `nn_rt.json`
