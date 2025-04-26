# Master Thesis Experiments for SID on Norwegian Dialects

This folder contains the code, data, and training scripts for Slot and Intent Detection (SID) experiments conducted as part of the master thesis.

---

## 🚀 Running Experiments
Experiments are run by submitting the provided SLURM scripts for fine-tuning SID models.


### Base Fine-Tuning
Fine-tune a pre-trained model on a SID dataset:
```bash
sbatch train.slurm <pre-trained model> <fine-tuning data>
sbatch train_with_dev.slurm <pre-trained model> <fine-tuning data>
```
`train_with_dev`: Used when `nomusic` is part of the fine-tuning dataset (i.e., needs a separate development set).

---

### Auxiliary Task Fine-Tuning
Fine-tuning with **one auxiliary task** (ner, dep, mlm, pos):
```bash
sbatch train_joint_aux.slurm <pre-trained model> <fine-tuning data> <aux task>
sbatch train_int_aux.slurm <pre-trained model> <fine-tuning data> <aux task>

sbatch train_joint_aux_with_dev.slurm <pre-trained model> <fine-tuning data> <aux task>
sbatch train_int_aux_with_dev.slurm <pre-trained model> <fine-tuning data> <aux task>
```
`train_joint_aux_with_dev` and `train_int_aux_with_dev`: Used when `nomusic` is part of the SID fine-tuning dataset.

Fine-tuning with **two auxiliary tasks**:
```bash
sbatch train_joint_two_aux_with_dev.slurm <pre-trained model> <fine-tuning data> <aux task 1> <aux task 2>
sbatch train_int_two_aux_with_dev.slurm <pre-trained model> <fine-tuning data> <aux task 1> <aux task 2>

```
NB: With intermediate task learning, the auxiliary tasks must be passed in the correct order.

---

### Arguments
Pre-trained models: MaChAmp requires a configuration file for the selected pre-trained model. Included model options:
- mbert, mdberta, mt0, rembert, xlmr, nbbert, norbert

#### Special note for NorBERT:
NorBERT uses **non-standard Huggingface code**, requiring small changes:
- For fine-tuning:  
    In `../machamp/machamp/model/machamp.py`, change:
    ```python
    self.mlm = AutoModel.from_pretrained(mlm)
    ```
    to:
    ```python
    self.mlm = AutoModel.from_pretrained(mlm, trust_remote_code=True)
    ```
- For prediction on Fox Educloud:  
    1. Create the directory (replace `ec-USERNAME` with your own username):
       ```
       /fp/homes01/u01/ec-USERNAME/.local/lib/python3.10/site-packages/transformers_modules/ltg/norbert3-base/
       ```      
    2. Clone the Huggingface NorBERT repo:
       ```
       https://huggingface.co/ltg/norbert3-base
       ```
    3. Move the cloned files into the created directory.  


Available Fine-tuning Datasets: Choose from files in the `data/` folder, except for files that include "dev" or "test" in their names.

---
### Model Outputs
Models, training logs, development predictions and evaluation metrics are saved under `machamp/logs`. Each model is fine-tuned **three times** with different seeds for robustness.


---
## 🔮 Prediction
To generate predictions across three trained runs:
```bash
sbatch predict.slurm <model_setup> <dataset>
```
Where <model_setup> is the name of the setup used in `machamp/logs`, and <dataset> is one of `dev`, `new_dev` (when model is fine-tuned on `nomusic`, or `test`.
When using joint auxiliary task learning setups, inside the prediction SLURM script, use: 
```bash
--dataset SID4LR
```

---

### Outputs
Predictions, overall and dialect-specific performance metrics for each run are saved individually under `machamp/logs/<model_setup>/predictions/`.
Mean and standard deviation across the three runs are saved under `predictions` in this folder, both for overall metrics and dialect-specific.
- Overall metrics = `.eval` files
- Dialect-specific metrics = `.dialect.eval` files
