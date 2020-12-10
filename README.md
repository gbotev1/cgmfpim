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
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#install-git-lfs">Install Git-LFS</a></li>
      <li><a href="#clone-repo">Clone Repo</a></li>
      <li><a href="#install-requirements">Install Requirements</a></li>
      <li><a href="#inflate-archives">Inflate Archives</a></li>
      <li><a href="#download-gcc-embeddings">Download GCC Embeddings</a></li>
    </ul>
  </li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#contact">Contact</a></li>
  <li><a href="#references">References</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project

[TODO]

<!-- GETTING STARTED -->
## Getting Started

The following steps should help you set up your environment:

### Install Git-LFS

This repository uses [Git Large File Storage](https://git-lfs.github.com), which should be downloaded, installed, and set-up for your user account before cloning our repository.

### Clone Repo

```sh
git clone https://github.com/gbotev1/cgmfpim.git
```

### Install Requirements

Our code is tested using Python 3.8. The provided [`requirements.txt`](requirements.txt) file delineates all requirements necessary to run any script in this repository. If you only plan on using our pre-computed archives, not all of these packages are necessary. Some scripts may necessitate the use of a GPU for which we require that an appropriate version of CUDA is installed. You should also make sure to install the [FAISS Library](https://github.com/facebookresearch/faiss) on your machine. We used the pre-compiled linux version from Anaconda with CUDA Toolkit 10.2 to enable GPU support.
```sh
pip3 install -r requirements.txt
```

### Inflate Archives

The following bash script is provided for convenience to easily extract the [`data.tsv`](data/11-25-20_21-1500.tsv.tar.bz2) file of scraped captions from [Imgflip](https://imgflip.com), the `meme_templates` directory of meme image templates into the `data` directory, and our custom [Google's Conceptual Captions (GCC) dataset](https://ai.google.com/research/ConceptualCaptions/download). Once extracted, the GCC dataset file [`gcc_full.tsv`](data/gcc_full.tsv.tar.bz2) we provide is nothing but a concatenation of the train and validation files available for download from the official linked dataset page after running each of the captions through [NLTK's Penn Treebank detokenizer](https://www.nltk.org/_modules/nltk/tokenize/treebank.html#TreebankWordDetokenizer). For the curious, this logic is defined in [`prepare_gcc.py`](prepare_gcc.py).
```sh
sh inflate_archives.sh
```

### Download GCC Embeddings
Along with the scripts that we used to generate these embeddings, we also provide a ready-to-use download of 2,841,059 2,048-dimensional embeddings for every image we could access from the [Google's Conceptual Captions (GCC) dataset](https://ai.google.com/research/ConceptualCaptions/download) training and validation splits. These embeddings were obtained from the output of the `avgpool` layer from the pre-trained [Wide ResNet-101-2](https://pytorch.org/docs/stable/torchvision/models.html#wide-resnet) on the ImageNet dataset.

<!-- CONTRIBUTING -->
## Contributing

Contributions are at the very essence of the open source community and are what keep projects alive and useful to the community that uses them. **We wholeheartedly welcome any and all contributions.**

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/DankFeature`)
3. Commit your Changes (`git commit -m 'Made memes more dank'`)
4. Push to the Branch (`git push origin feature/DankFeature`)
5. Open a Pull Request

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

A collection of papers from which we took inspiration:
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
