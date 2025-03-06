# Movie Recommender System

This is a Python-based movie recommender system that uses user-based collaborative filtering to recommend movies based on ratings they have given. The system utilises a dataset of 100,000+ ratings, by 600+ different users for over 9000+ different movies. This data exists in a MongoDB database that the system interacts with.

## Prerequisites

Before you begin, please make sure you have the following installed:
- Python 3.12 (Specifically this version)
- pip (Python’s package installer)

You can install Python 3.12 using (Mac):
```
brew install python@3.12
```
(Assuming you have homebrew installed)

Or on Windows via the installation page: https://www.python.org/downloads/


## Setup Instructions

### 1. It is recommended to setup a python virtual environment where you install all the required dependencies:

#### First create a venv in Python3.12:
```
python3.12 -m venv your_venv_name
```

#### Activate the venv (Mac):

```
source your_venv_name/bin/activate
```

#### Activate the venv (Windows):
```
your_venv_name\Scripts\activate
```

### 2. Install the required dependencies from the requirements.txt file:

```
pip install -r requirements.txt
```

### 3. Run the script to generate recommendations:

```
python3.12 main.py
```