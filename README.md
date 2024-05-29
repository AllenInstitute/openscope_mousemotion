## Installation

### Dependencies:

- Windows OS (see **Camstim package**)
- python 2.7
- psychopy 1.82.01
- camstim 0.2.4

### Installation with [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

1. Navigate to repository and install conda environment.  
    `conda env create -f environment.yml`
2. Activate the environment.  
    `conda activate allen_stimulus`
3. Install the AIBS `camstim` package in the environment.  
    `pip install camstim/.`
4. Download required video clips from [movie_clips.zip](https://tigress-web.princeton.edu/~dmturner/allen_stimulus/movie_clips.zip)
   Extract into the `data` directory.
5. Run the script:
    `python pilot_script.py`
