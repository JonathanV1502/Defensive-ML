# Defensive-ML

Intrusion detection systems (IDS) are designed to monitor network or system activities for malicious actions and policy violations ensuring the security of a network. Machine learning enhances the capabilities of intrusion detection by enabling the system to learn from data without explicit programming. Machine learning algorithms analyze vast amounts of network traffic data to detect patterns and anomalies. To keep pace with evolving cyberthreats; these algorithms adapt over time, improving their accuracy in identifying new and evolving threats.

However, these IDS using deep learning methods introduce new vulnerabilities. For example, in this repository, we show how the use of a fast gradient sign method (FGSM) attack can help attackers understand how to craft their attacks in a way that would  "camouflage" their efforts from the IDS. We further show how the FGSM is less effective against ensemble models.



## Set-up
### Dependences
``conda create --name DF-ML --file spec-file.txt``

### Loading Dataset
``python data.py``

``python preprocessing/cicids2017.py``
