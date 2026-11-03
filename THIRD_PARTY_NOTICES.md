# Third-party notices

The source ZIP does not vendor upstream model packages or weights. The local
BRITS execution helper adapts the PyPOTS 1.5 BRITS/RITS forward equations; its
upstream BSD-3-Clause notice is retained in
[`docs/licenses/pypots-bsd-3-clause.txt`](docs/licenses/pypots-bsd-3-clause.txt).
Installed packages and downloaded local checkpoints retain their own terms; the project's MIT
license applies to its original code. Exact sources and checkpoint hashes are
recorded in `model_registry.yaml` and in local execution artifacts.

The native server integrations use PyPOTS 1.5 (BSD-3-Clause), pinned official T1
(MIT), official MOMENT source version 0.1.5 at commit
`38f7310ad594100747ca2a8357e9c7ca7d323e0e` (MIT), and TS-ICL 0.2.0 at commit
`6d4861e9eb29d59f05a0c253432afb2ae936e02a` (TS-ICL Non-Commercial License v1.0,
code license revised June 9, 2026). The pinned TS-ICL checkpoint carries its own
copy of that named license, revised June 30, 2026. Keep both original license texts.
They include restrictions on commercial/production use and outputs. Installing
these packages into the same cuda_env does not change their licenses or license
their artifacts under MIT.

BRITS, SAITS, CSDI, ImputeFormer and OpenFIM upstream repositories identify MIT;
this does not automatically establish a license for every external checkpoint.
The FIM checkpoint licenses remain unresolved. CopyPasteImputation identifies
LGPL-3.0; `cpi_energy_donor_adapted` is separately named original code, not copied
LGPL source. The inspected R24 TSFM_Imputation repository has no asserted open
license; its files are not redistributed as MIT.

Primary sources: [PyPOTS](https://github.com/WenjieDu/PyPOTS),
[MOMENT source](https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e),
[TS-ICL code license](https://github.com/EDF-Lab/ts-icl/blob/6d4861e9eb29d59f05a0c253432afb2ae936e02a/LICENSE),
[TS-ICL checkpoint license](https://huggingface.co/taharnbl/TS-ICL/blob/19c94031439fb31f36ce395088ee50a6762d3774/LICENSE),
[CPI](https://github.com/KIT-IAI/CopyPasteImputation),
[R24](https://github.com/taharnbl/tsfm_imputation).

The UCI ElectricityLoadDiagrams data are CC BY 4.0 according to their official
record. The private H1/H2/I1 data have no established public redistribution
license. Neither dataset inherits this repository's MIT license.
