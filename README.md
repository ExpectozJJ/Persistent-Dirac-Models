# Persistent Dirac for Molecular Representation

This manual is for the code implementation of paper "Persistent Dirac for Molecular Representation".

# Code Requirements
---
        Platform: Python>=3.6, MATLAB 2016B
        Python Packages needed: math, numpy>=1.19.5, scipy>=1.4.1, scikit-learn>=0.20.3, GUDHI 3.0.0

# Details about each step

## Persistent Dirac Models for OIHP classification 
Before the representation, the atom coordinates from each frame needs to extracted accordingly to the atom subsets CHNPbX and CNPbX. 
The atom coordinates stored in *.txt files for each frame can be found in the Dirac_OIHP.zip.  

For each frame, we construct the simplicial complexes to generate the Dirac matrices. 
```python
Dirac_CHNPbX.py --> Construct and generate CHNPbX Persistent Dirac features. 
Dirac_CNPbX.py --> Construct and generate CNPbX Persistent Dirac features.
Dirac_classify.py --> Classify 9 types of OIHP using CHNPbX and CNPbX Persistent Dirac features.
Dirac_classify_CNPbX.py --> Classify 9 types of OIHP using CNPbX Persistent Dirac features.
```

## Weighted Dirac Matrices
---
For a guanine amino acid at 1.2 angstroms, we construct its simplicial complex to construct its weighted and unweighted Dirac matrices. 
```python
weighted_Dirac_guanine.py --> Replicates the images on Figure 3. 
```

## Cite
If you use this code in your research, please cite our paper:

* JunJie Wee, Ginestra Bianconi, and Kelin Xia. "Persistent Dirac for molecular representation." arXiv preprint arXiv:2302.02386 (2023).
