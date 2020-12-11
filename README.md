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
    <img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg?style=for-the-badge"/>
    <img alt="repo size" src="https://img.shields.io/github/repo-size/gbotev1/cgmfpim?style=for-the-badge"/>
    <img alt="total lines" src="https://img.shields.io/tokei/lines/github/gbotev1/cgmfpim?style=for-the-badge"/>
    <img alt="AGPL-3.0" src="https://img.shields.io/github/license/gbotev1/cgmfpim?style=for-the-badge"/>
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
  <li><a href="#references">References</a></li>
</ul>

<!-- ABOUT THE PROJECT -->
## About the Project

[TODO]

<!-- GETTING STARTED -->
## Getting Started

The following steps should help you set up your environment:

### 1. Install Git-LFS

This repository uses [Git Large File Storage](https://git-lfs.github.com), which should be downloaded, installed, and set up for your user account before cloning our repository.

### 2. Clone repository

```sh
git clone https://github.com/gbotev1/cgmfpim.git
```

### 3. Install requirements

Our code is tested using Python 3.8. The provided [`requirements.txt`](requirements.txt) file delineates all requirements necessary to run any script in this repository. If you only plan on using our pre-computed archives, not all of these packages are necessary. Some scripts may necessitate the use of a GPU for which we require that an appropriate version of CUDA is installed. You should also make sure to install the [FAISS Library](https://github.com/facebookresearch/faiss) on your machine. We used the pre-compiled linux version from Anaconda with CUDA Toolkit 10.2 to enable GPU support.
```sh
pip3 install -r requirements.txt
```

### 4. Inflate archives

The [following `sh` script](inflate_archives.sh) is provided for convenience to easily extract `meme_data.tsv` and `meme_data_top.tsv` from [this file](data/11-25-20_21-1500.tsv.tar.gz) of scraped captions from [Imgflip](https://imgflip.com) on November 25, 2020 as well as a `meme_templates` directory of meme image templates into the [`data`](data) directory. The `meme_data_top.tsv` file is a filtered version of the full `meme_data.tsv`, where at most the top 100 meme captions by number of upvotes are saved. It also extracts our custom [Google's Conceptual Captions (GCC) dataset](https://ai.google.com/research/ConceptualCaptions/download). Once extracted, the GCC dataset file [`gcc_full.tsv`](data/gcc_full.tsv.tar.gz) we provide is nothing but a concatenation of the train and validation files available for download from the official linked dataset page after running each of the captions through [NLTK's Penn Treebank detokenizer](https://www.nltk.org/_modules/nltk/tokenize/treebank.html#TreebankWordDetokenizer). For those curious, this logic is defined in [`prepare_gcc.py`](prepare_gcc.py).
```sh
sh inflate_archives.sh
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

**Libraries**
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

**Papers**
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
