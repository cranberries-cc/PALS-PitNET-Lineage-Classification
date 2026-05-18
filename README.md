<div align="center">

# PLAS — Pituitary Lineage Analysis System

**Lineage Classification of Pituitary Neuroendocrine Tumors from Whole-Slide Images  
using Attention-Guided Graph Representation Learning**

[![Paper](https://img.shields.io/badge/Paper-Endocrine%20Pathology-blue)](https://link.springer.com/article/10.1007/s12022-026-09919-x)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs12022--026--09919--x-blue)](https://doi.org/10.1007/s12022-026-09919-x)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](#)

<img src="workflows.png" width="90%">

</div>

---

## 📰 News

- **[2026/05/07]** Full scripts coming soon.
- **[2026/05/04]** Accepted by *Endocrine Pathology* 🎉

## 📖 Abstract

Pituitary neuroendocrine tumors (PitNETs) are common sellar neoplasms and represent a major component of routine pituitary pathology. In the 2022 World Health Organization (WHO) Classification, transcription factor-defined lineage assignment is central to diagnosis. However, lineage-related morphologic information on routine hematoxylin and eosin (H&E)-stained whole-slide images (WSIs) has not been systematically characterized. We developed an attention-guided graph neural network to predict PitNET lineage directly from H&E-stained WSIs and to identify regions prioritized for classification. Consecutive surgically PitNETs from Beijing Tiantan Hospital (2021-2025) were included. In the internal hold-out set, the five-fold cross-validation models achieved a mean F1-score of **92.78%** and a mean balanced accuracy of **94.84%**. In a temporally independent validation cohort, the final model achieved an F1-score of **87.64%** and a balanced accuracy of **89.48%**. Performance varied across lineages and histomorphologic subtypes, and the predominant error pattern was misassignment of PIT-1 and T-PIT tumors to SF-1. Attention maps predominantly highlighted tumor-rich regions. Quantitative cell-level morphometry supported lineage-associated patterns, including larger cell and nuclear size in PIT-1 tumors, more elongated nuclei in SF-1 tumors, and higher cellular density with reduced intercellular spacing in T-PIT tumors. In six cases with multiple synchronous PitNETs of distinct lineages, patch-level prediction maps corresponded closely to transcription factor immunohistochemistry. In a small exploratory subset with available DNA methylation data, methylation-based classification was more concordant with model predictions than with routine clinical diagnosis. These findings indicate that routine H&E-stained WSIs contain learnable morphologic information related to PitNET lineage and that attention-guided spatial modeling provides an interpretable framework for characterizing lineage-associated patterns in PitNETs.

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{hao2026lineage,
  author  = {Hao, J. and Wang, C. and Li, J. and Du, J. and Qian, Q. and Liu, X.},
  title   = {Lineage Classification of Pituitary Neuroendocrine Tumors From Whole-Slide Images Using Attention-Guided Graph Representation Learning},
  journal = {Endocrine Pathology},
  volume  = {37},
  number  = {1},
  pages   = {21},
  year    = {2026},
  doi     = {10.1007/s12022-026-09919-x},
  url     = {https://doi.org/10.1007/s12022-026-09919-x}
}
```

## 📧 Contact

For questions, please contact [chenwang020916@gmail.com](mailto:chenwang020916@gmail.com).

## 📄 License

This project is released under the [MIT License](LICENSE).

