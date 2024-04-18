# Defensive-ML


However, these IDS using deep learning methods introduce new vulnerabilities. For example, in this repository, we show how the use of a fast gradient sign method (FGSM) attack can help attackers understand how to craft their attacks in a way that would  "camouflage" their efforts from the IDS. We further show how the FGSM is less effective against ensemble models.



## Set-up
### Dependences
``conda create --name DF-ML --file spec-file.txt``

### Loading Dataset
``python data.py``

``python preprocessing/cicids2017.py``