# Data Licenses and Attribution

This directory contains two categories of data, each of which is distributed under a different license.

## `data/tasks/`: molecular property data sets

The curated data sets in `data/tasks/` are derived from the four sources listed below.
Each data set is distributed under its original license, as shown in the table.
Note that commercial use is not permitted for the Density data set (CC BY-NC 3.0).

### Individual sources

| Task | File | Original source | Original license |
|------|------|-----------------|------------------|
| Density | `tasks/density/curated_data.csv` | Pandey and Roy (2024), *Energy Adv.*, [doi.org/10.1039/D4YA00215F](https://doi.org/10.1039/D4YA00215F) | CC BY-NC 3.0 |
| LogD | `tasks/logd/curated_data.csv` | AstraZeneca data set deposited in ChEMBL (CHEMBL3301361), [doi.org/10.6019/CHEMBL3301361](https://doi.org/10.6019/CHEMBL3301361), distributed as the Lipophilicity data set of MoleculeNet (Wu et al., 2018) | CC BY-SA 3.0 |
| LogS | `tasks/logs/curated_data.csv` | AqSolDB, Sorkun et al. (2019), [doi.org/10.7910/DVN/OVHAW8](https://doi.org/10.7910/DVN/OVHAW8) | CC0 1.0 |
| Melting point | `tasks/mp/curated_data.csv` | Jean-Claude Bradley Open Melting Point Dataset, Bradley et al. (2014), [doi.org/10.6084/m9.figshare.1031637](https://doi.org/10.6084/m9.figshare.1031637) | CC0 1.0 |

## `data/transformation_rules/`: MMP transformation rules

This directory contains matched molecular pair (MMP) transformation rules derived from
**ChEMBL 31**, ranked by the frequency with which each transformation occurs in that database.

| File | Rules | Frequency range | Used for |
|------|-------|-----------------|----------|
| `transformation_rules_top1k.csv` | 1,000 | 42 – 1,635 | The augmentation used throughout this work |

This file is redistributed under **Creative Commons Attribution-ShareAlike 3.0
Unported (CC BY-SA 3.0)**, which is the license of the ChEMBL data. The logD data set in
`data/tasks/logd/` is also derived from ChEMBL and is redistributed under the same license.

Full license text: https://creativecommons.org/licenses/by-sa/3.0/

**Attribution:** ChEMBL Database, European Molecular Biology Laboratory –
European Bioinformatics Institute (EMBL-EBI).
https://www.ebi.ac.uk/chembl/
