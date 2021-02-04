
# Crowdsourcing More Effective Initializations for Single-Target Trackers through Automatic Re-querying
This repository contains the code used for the paper "Crowdsourcing More Effective Initializations for Single-target Trackers through Automatic Re-querying," to be published at the 2021 Conference on Human Factors in Computing Systems (CHI). If you find this work helpful, please cite:

    @inproceedings{lemmer_crowdsourcing_2021,
	address = {Virtual (Originally Yokohama, Japan)},
	title = {Crowdsourcing {More} {Effective} {Initializations} for {Single}-{Target} {Trackers} through {Automatic} {Re}-querying},
	booktitle = {Proceedings of the 2021 {Conference} on {Human} {Factors} in {Computing} {Systems}},
	publisher = {ACM Press},
	author = {Lemmer, Stephan J. and Song, Jean Y. and Corso, Jason J.},
	month = may,
	year = {2021}
}

![Smart Replacement Teaser Figure](https://github.com/lemmersj/crowdsourcing-effective-initializations/blob/main/teaser_fig.png)
This work is separated into two folders: /mturk_tools/ contains files related to the crowdsourcing of initialization bounding boxes, and is derived from the annotation tool [here](https://github.com/kyamagu/bbox-annotator).  /tracker/ contains the files related to the analysis of tracker performance using the crowdsourced initializations. Its code is derived from [DaSiamRPN](https://github.com/foolwood/DaSiamRPN). License information for each folder is available in that folder, as the licenses are derived from those of the original work (the annotation tool is BSD 3-Clause, while DaSiamRPN is MIT). Original code in this repository is under the MIT license.
