# PFC Self-Administration
### A pipeline analysis of the prefrontal cortex region in mice under various reinstatement tests.

*Code by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=for-the-badge&logo=github)](https://github.com/thejoshbq)
[![](https://img.shields.io/badge/@thejoshbq-grey?style=for-the-badge&logo=X)](https://x.com/thejoshbq)
[![](https://img.shields.io/badge/Otis_Lab-grey?style=for-the-badge)](https://www.otis-lab.org/)

---

## Overview

This repository serves as an acquisition tool and analysis pipeline for the extracted neural-imaging data and corresponding behavioral data.
Within the repository, you will find the following:

- `data` - contains the extracted data sorted by day > animal > FOV
- `src` - contains utility modules used in the analysis pipeline
- `main.py` - the main analysis script

> Note: This analysis requires the `NSync2P` package, which you can find and install [here](https://github.com/otis-lab-musc/nsync2p).

## Using Pipeline

To use this repository on your local machine:

- **Clone the repository** - `git clone https://github.com/thejoshbq/PFC_Self-Admin_Analysis.git`
- **Install the dependencies** - `pip install -r requirements.txt`
- **Run the analysis** - `python3 main.py`

<br><br>
<div align="center">
  <h2>Copyright & License</h2>
  <p>© 2025 <a href="http://www.otis-lab.org">Otis Lab</a>. All rights reserved.</p>
  <p>This project is licensed under the <a href=""><strong>LICENSE</strong></a>.</p>
  <p>For more information, please contact the author at <a href="mailto:thejoshbq@proton.me"><i>thejoshbq@proton.me</i></a>
</div>

