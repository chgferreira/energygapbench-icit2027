# Reference inventory

This inventory preserves the R1–R28 identifiers from the initial directed literature
and implementation audit. Links identify sources considered during development;
inclusion does not imply integration, reproduction, superiority, or a systematic
review. It is not a fresh verification of publication/license metadata. Check the
cited version, authors, venue/status, and applicable terms before manuscript release.

| ID | Source | Role in the original design |
| --- | --- | --- |
| R1 | [BRITS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html); [code](https://github.com/caow13/BRITS) | Recurrent imputation reference |
| R2 | [SAITS](https://arxiv.org/abs/2202.08516); [code](https://github.com/WenjieDu/SAITS) | Self-attention imputation reference; DOI 10.1016/j.eswa.2023.119619 |
| R3 | [CSDI](https://proceedings.neurips.cc/paper_files/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html); [code](https://github.com/ermongroup/CSDI) | Conditional diffusion/joint samples |
| R4 | [ImputeFormer](https://doi.org/10.1145/3637528.3671751); [code](https://github.com/tongnie/ImputeFormer) | Low-rank multivariate imputation |
| R5 | [T1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41ff2b921a875902fad0c1840593c19a-Abstract-Conference.html); [code](https://github.com/Oppenheimerdinger/T1) | Channel/head binding and explicit D=1 audit |
| R6 | [MOMENT](https://proceedings.mlr.press/v235/goswami24a.html); [code](https://github.com/moment-timeseries-foundation-model/moment); [checkpoint](https://huggingface.co/AutonLab/MOMENT-1-base) | Frozen pretrained masked reconstruction |
| R7 | [Foundation Inference Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d03791377a31cb3e7357014ba58eb80-Abstract-Conference.html); [code](https://github.com/FIM4Science/OpenFIM) | Temporal/pointwise variant and checkpoint audit |
| R8 | [Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios](https://ojs.aaai.org/index.php/AAAI/article/view/33931); [code](https://github.com/rafid009/SADI_official_repository) | Partial-blackout context; terms require verification |
| R9 | [TS-ICL](https://arxiv.org/abs/2606.05878); [code](https://github.com/EDF-Lab/ts-icl) | Time-indexed pretrained imputation; noncommercial terms |
| R10 | [Time-Gated Multi-Scale Flow Matching for Time-Series Imputation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/51d317df78eded9eb3c9d3fb1091c279-Abstract-Conference.html) | Further comparison candidate; not reproduced here |
| R11 | [TSI-Bench](https://arxiv.org/abs/2406.12747); [source collection](https://github.com/WenjieDu/Awesome_Imputation) | Unified evaluation and downstream/resource context |
| R12 | [ImputeGAP](https://arxiv.org/abs/2503.15250); [code](https://github.com/eXascaleInfolab/ImputeGAP) | Modular evaluation and missingness-generation context |
| R13 | [Benchmarking Missing Data Imputation Methods for Time Series Using Real-World Test Cases](https://proceedings.mlr.press/v287/toye25a.html) | Realistic mechanisms and strong simple references |
| R14 | [Clinical time-series benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC13639038/) | Structured missingness, temporal separation, downstream evaluation; supplied DOI 10.1038/s41467-026-77303-8 |
| R15 | [Bridging Smart Meter Gaps](https://arxiv.org/abs/2501.07276) | Direct smart-meter benchmark precedent |
| R16 | [Data-Driven Copy-Paste Imputation for Energy Time Series](https://arxiv.org/abs/2101.01423); [code](https://github.com/KIT-IAI/CopyPasteImputation) | Cumulative-energy donor precedent; separately named adaptation here |
| R17 | [Data-Driven Power Time Series Imputation Using Cumulative Pattern Matching](https://perfiles.ibero.mx/en/publications/data-driven-power-time-series-imputation-using-cumulative-pattern-2/); [DOI](https://doi.org/10.1007/978-3-032-09044-7_12) | Boundary-energy and cumulative-pattern precedent |
| R18 | [The Jackknife and the Bootstrap for General Stationary Observations](https://doi.org/10.1214/aos/1176347265) | Dependent-data resampling foundations |
| R19 | [Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/v7/demsar06a.html) | Independent-unit interpretation; seeds/windows are not datasets |
| R20 | [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | Probabilistic scoring and calibration/sharpness context |
| R21 | [A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733) | Holm multiplicity adjustment |
| R22 | [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014); [DOI](https://doi.org/10.24432/C58C86) | Optional external data source and its own access/clock conventions |
| R23 | [False discovery rate under dependency](https://doi.org/10.1214/aos/1013699998) | Predefined exploratory BY families |
| R24 | [Are Time-Indexed Foundation Models the Future of Time Series Imputation?](https://arxiv.org/abs/2511.05980); [code](https://github.com/taharnbl/tsfm_imputation) | TabPFN-TS/MoTM implementation and artifact audit |
| R25 | [Rethinking Time-Series Imputation as Conditional Inference along Temporal Evolution](https://proceedings.mlr.press/v306/fan26h.html) | CBiT compressed-history reference; implementation not verified here |
| R26 | [tqdm API](https://tqdm.github.io/docs/tqdm/) | Presentation of progress, not durable accounting |
| R27 | [PyTorch saving/loading guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | Native serialization; additional RNG/sampler/durability guarantees require tests |
| R28 | [Optuna RDB persistence](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html) | Original persistence reference; current HPO uses a fixed grid, not an Optuna study |

Primary component revisions, checkpoint digests, and implementation/availability
statuses belong to `model_registry.yaml` and run artifacts. A paper/source link is
not a license grant for every checkpoint. Further novelty, bibliography, and
publication-status review remains TODO in [development status](development_status.md).
