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
    <img alt="GPL-3.0" src="https://img.shields.io/github/license/gbotev1/cgmfpim?style=for-the-badge"/>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ol>
  <li><a href="#about-the-project">About The Project</a></li>
  <li>
    <a href="#getting-started">Getting Started</a>
    <ul>
      <li><a href="#clone-repo">Clone Repo</a></li>
      <li><a href="#install-requirements">Install Requirements</a></li>
      <li><a href="#inflate-archives">Inflate Archives</a></li>
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

### Clone Repo

```sh
git clone https://github.com/gbotev1/cgmfpim.git
```

### Install Requirements

Our code is tested using Python 3.8.
```sh
pip3 install -r requirements.txt
```

### Inflate Archives

The following bash script is provided for convenience to extract the `captions.tsv` file of scraped captions from [Imgflip](https://imgflip.com) and `meme_templates` directory of meme templates easily.
```sh
sh inflate_archives.sh
```

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
Distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for more information.

<!-- CONTACT -->
## Contact

Listed in alphabetical order by last name:
* Georgie Botev - gbotev1@jhu.edu
	* Pursuing Masters in Computer Science at Johns Hopkins University
* Peter Ge - yge15@jhmi.edu
  * TODO
* Samantha Zarate - slzarate@jhu.edu
  * TODO

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
