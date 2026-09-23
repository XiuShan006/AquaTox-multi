# Uni-Mol dependencies and external weights

The main AquaTox-multi model does not require Uni-Mol. The optional Uni-Mol comparison requires Uni-Core 0.0.1 and the dependencies in `requirements.txt`.

The Uni-Mol control requires the official pretrained checkpoint at:

`weights/mol_pre_no_h_220816.pt`

Download it from:

<https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt>


The fine-tuned comparison workflow and its configuration are under [`src/unimol_control/`](../src/unimol_control/). Download the file before running that workflow; it is intentionally not redistributed here.
