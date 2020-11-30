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
    <a href="https://github.com/badges/shields/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/badges/shields" /></a>
    <a href="#backers" alt="Backers on Open Collective">
        <img src="https://img.shields.io/opencollective/backers/shields" /></a>
    <a href="#sponsors" alt="Sponsors on Open Collective">
        <img src="https://img.shields.io/opencollective/sponsors/shields" /></a>
    <a href="https://github.com/badges/shields/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/badges/shields" /></a>
    <a href="https://circleci.com/gh/badges/shields/tree/master">
        <img src="https://img.shields.io/circleci/project/github/badges/shields/master" alt="build status"></a>
    <a href="https://circleci.com/gh/badges/daily-tests">
        <img src="https://img.shields.io/circleci/project/github/badges/daily-tests?label=service%20tests"
            alt="service-test status"></a>
    <a href="https://coveralls.io/github/badges/shields">
        <img src="https://img.shields.io/coveralls/github/badges/shields"
            alt="coverage"></a>
    <a href="https://lgtm.com/projects/g/badges/shields/alerts/">
        <img src="https://img.shields.io/lgtm/alerts/g/badges/shields"
            alt="Total alerts"/></a>
    <a href="https://github.com/badges/shields/compare/gh-pages...master">
        <img src="https://img.shields.io/github/commits-since/badges/shields/gh-pages?label=commits%20to%20be%20deployed"
            alt="commits to be deployed"></a>
    <a href="https://discord.gg/HjJCwm5">
        <img src="https://img.shields.io/discord/308323056592486420?logo=discord"
            alt="chat on Discord"></a>
    <a href="https://twitter.com/intent/follow?screen_name=shields_io">
        <img src="https://img.shields.io/twitter/follow/shields_io?style=social&logo=twitter"
            alt="follow on Twitter"></a>
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

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)<br/>
Distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for more information.

<!-- CONTACT -->
## Contact

Listed in alphabetical order by last name:
* Georgie Botev - gbotev1@jhu.edu
	* Pursuing Masters in Computer Science at Johns Hopkins University
* Peter Ge - yge15@jhmi.edu
* Samantha Zarate - slzarate@jhu.edu

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
