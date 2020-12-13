<!-- INFO -->
<p align="center">
  <h3 align="center">Computer-Generated Memes for Plugged-In Machines</h3>
  <p align="center">
    Using AI, machine learning, and NLP to generate memes.
    <br/>
  </p>
</p>

<!-- SHIELDS -->
<p align="center">
    <img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg?color=blue&style=for-the-badge"/>
    <img alt="repo size" src="https://img.shields.io/github/repo-size/gbotev1/cgmfpim?color=blue&style=for-the-badge"/>
    <img alt="total lines" src="https://img.shields.io/tokei/lines/github/gbotev1/cgmfpim?color=blue&style=for-the-badge"/>
    <img alt="AGPL-3.0" src="https://img.shields.io/github/license/gbotev1/cgmfpim?color=blue&style=for-the-badge"/>
    <img alt="beta mode" src="https://forthebadge.com/images/badges/60-percent-of-the-time-works-every-time.svg"/>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ul>
	<li><a href="#about-the-project">About the Project</a></li>
	<li>
		<a href="#getting-started">Getting Started</a>
		<ol>
			<li><a href="#1-install-git-lfs">Install Git-LFS</a></li>
			<li><a href="#2-clone-repository">Clone repository</a></li>
			<li><a href="#3-install-requirements">Install requirements</a></li>
			<li><a href="#4-inflate-archives">Inflate archives</a></li>
			<li><a href="#5-download-gcc-embeddings">Download GCC embeddings</a></li>
		</ol>
	</li>
	<li><a href="#contributing">Contributing</a></li>
	<li><a href="#license">License</a></li>
	<li><a href="#contact">Contact</a></li>
  	<li>
		<a href="#references">References</a>
		<ol>
			<li><a href="#code">Code</a></li>
			<li><a href="#papers">Papers</a></li>
		</ol>
	</li>
</ul>

<!-- ABOUT THE PROJECT -->
## About the Project

This project is still under active development, and should be understood to be in **beta** mode.

<!-- GETTING STARTED -->
## Getting Started

The following steps should help you set up your environment:

### 1. Install Git-LFS

This repository uses [Git Large File Storage](https://git-lfs.github.com), which should be downloaded and installed before cloning our repository. In particular, make sure to run the following command before cloning:
```sh
git lfs install
```

### 2. Clone repository

```sh
git clone https://github.com/gbotev1/cgmfpim.git
```

### 3. Install requirements

Our code is tested using Python 3.8. The provided [`requirements.txt`](requirements.txt) file delineates all requirements necessary to run any script in this repository. If you plan on using only our pre-computed archives, then not all of these packages are necessary. Some scripts may necessitate the use of a GPU for which an appropriate version of CUDA must be installed. You should also make sure to install the [FAISS Library](https://github.com/facebookresearch/faiss) on your machine. We used the pre-compiled Linux version from Anaconda with CUDA Toolkit 10.2 for our experiments.
```sh
pip3 install -U -r requirements.txt
```
**Note:**
> The pytorch-lightning['extra'] PyPI package seems to not correctly install the extra dependencies, so we have modified our [`requirements.txt`](requirements.txt) file to manually install the [Horovod](https://github.com/horovod/horovod) and [FairScale](https://github.com/facebookresearch/fairscale) PyPI packages.

### 4. Inflate archives

The [following `bash` script](inflate_archives.sh) is provided for convenience to easily extract `meme_data.tsv` and `meme_data_top.tsv` from [this file](data/11-25-20_21-1500.tsv.tar.gz) of scraped captions from [Imgflip](https://imgflip.com) on November 25, 2020 as well as a `meme_templates` directory of meme image templates into the [`data`](data) directory. The `meme_data_top.tsv` file is a filtered version of the full `meme_data.tsv`, where at most the top 100 meme captions by number of upvotes are saved. It also extracts our custom [Google's Conceptual Captions (GCC) dataset](https://ai.google.com/research/ConceptualCaptions/download). Once extracted, the GCC dataset file [`gcc_full.tsv`](data/gcc_full.tsv.tar.gz) we provide is nothing but a concatenation of the train and validation files available for download from the official linked dataset page after running each of the captions through [NLTK's Penn Treebank detokenizer](https://www.nltk.org/_modules/nltk/tokenize/treebank.html#TreebankWordDetokenizer). For those curious, this logic is defined in [`prepare_gcc.py`](prepare_gcc.py).
```sh
inflate_archives.sh
```

### 5. Download GCC embeddings
Along with the scripts that we used to generate these embeddings, we also provide a ready-to-use download of 2,841,059 2,048-dimensional embeddings for every image we could access from the [Google's Conceptual Captions (GCC) dataset](https://ai.google.com/research/ConceptualCaptions/download) training and validation splits. These embeddings were obtained from the output of the `avgpool` layer from the pre-trained [Wide ResNet-101-2](https://pytorch.org/docs/stable/torchvision/models.html#wide-resnet) on the ImageNet dataset.

<!-- CONTRIBUTING -->
## Contributing

Contributions are at the very essence of the open source community and are what keep projects alive and useful to the community that uses them. **We wholeheartedly welcome any and all contributions.**

1. Fork the project
2. Create your feature branch (`git checkout -b feature/DankFeature`)
3. Commit your changes (`git commit -m 'Made memes more dank'`)
4. Push to the branch (`git push origin feature/DankFeature`)
5. Open a pull request

<!-- LICENSE -->
## License
Distributed under the GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE) for more information.

<!-- CONTACT -->
## Contact

Listed in alphabetical order by last name:
* Georgie Botev - gbotev1@jhu.edu
	* Pursuing Masters in Computer Science at Johns Hopkins University
* Peter Ge - yge15@jhmi.edu
  * Pursing PhD in Biomedical Engineering at Johns Hopkins University
* Samantha Zarate - slzarate@jhu.edu
  * Pursuing PhD in Computer Science at Johns Hopkins University

<!-- REFERENCES -->
## References

### Code
* [Huggingface's 🤗Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.](https://github.com/huggingface/transformers)
	```
	@inproceedings{wolf-etal-2020-transformers,
	    title = "Transformers: State-of-the-Art Natural Language Processing",
	    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
	    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
	    month = oct,
	    year = "2020",
	    address = "Online",
	    publisher = "Association for Computational Linguistics",
	    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
	    pages = "38--45"
	}
	```
* [PyTorch Lightning: The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.](https://github.com/PyTorchLightning/pytorch-lightning)
	```
	@article{falcon2019pytorch,
	  title={PyTorch Lightning},
	  author={Falcon, WA},
	  journal={GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning},
	  volume={3},
	  year={2019}
	}
	```
* [FAISS: A library for efficient similarity search and clustering of dense vectors.](https://github.com/facebookresearch/faiss)
	```
	@article{JDH17,
	  title={Billion-scale similarity search with GPUs},
	  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
	  journal={arXiv preprint arXiv:1702.08734},
	  year={2017}
	}
	```
* [FairScale: PyTorch extensions for high performance and large scale training.](https://github.com/facebookresearch/fairscale)
	```
	@misc{kim2020torchgpipe,
	      title={torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models}, 
	      author={Chiheon Kim and Heungsub Lee and Myungryong Jeong and Woonhyuk Baek and Boogeon Yoon and Ildoo Kim and Sungbin Lim and Sungwoong Kim},
	      year={2020},
	      eprint={2004.09910},
	      archivePrefix={arXiv},
	      primaryClass={cs.DC}
	}
	```
	```
	@misc{rajbhandari2020zero,
	      title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models}, 
	      author={Samyam Rajbhandari and Jeff Rasley and Olatunji Ruwase and Yuxiong He},
	      year={2020},
	      eprint={1910.02054},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
	```
	```
	@misc{shoeybi2020megatronlm,
	      title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism}, 
	      author={Mohammad Shoeybi and Mostofa Patwary and Raul Puri and Patrick LeGresley and Jared Casper and Bryan Catanzaro},
	      year={2020},
	      eprint={1909.08053},
	      archivePrefix={arXiv},
	      primaryClass={cs.CL}
	}
	```
	```
	@misc{johnson2020adascale,
	      title={AdaScale SGD: A User-Friendly Algorithm for Distributed Training}, 
	      author={Tyler B. Johnson and Pulkit Agrawal and Haijie Gu and Carlos Guestrin},
	      year={2020},
	      eprint={2007.05105},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}
	}
	```
* [Horovod: Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.](https://github.com/horovod/horovod)
	```
	@article{sergeev2018horovod,
	  Author = {Alexander Sergeev and Mike Del Balso},
	  Journal = {arXiv preprint arXiv:1802.05799},
	  Title = {Horovod: fast and easy distributed deep learning in {TensorFlow}},
	  Year = {2018}
	}
	```

### Papers
* [Dank Learning: Generating Memes Using Deep Neural Networks](https://arxiv.org/pdf/1806.04510.pdf)
	```
	@misc{peirson2018dank,
	      title={Dank Learning: Generating Memes Using Deep Neural Networks}, 
	      author={Abel L Peirson V au2 and E Meltem Tolunay},
	      year={2018},
	      eprint={1806.04510},
	      archivePrefix={arXiv},
	      primaryClass={cs.CL}
	}
	```
* [Language Models are Unsupervised Multitask Learners](http://www.persagen.com/files/misc/radford2019language.pdf)
	```
	@article{radford2019language,
	    title={Language Models are Unsupervised Multitask Learners},
	    author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
	    year={2019}
	  }
	```
