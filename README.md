# Defensive-ML

Intrusion detection systems (IDS) are designed to monitor network or system activities for malicious actions and policy violations ensuring the security of a network. Machine learning enhances the capabilities of intrusion detection by enabling the system to learn from data without explicit programming. Machine learning algorithms analyze vast amounts of network traffic data to detect patterns and anomalies. To keep pace with evolving cyberthreats; these algorithms adapt over time, improving their accuracy in identifying new and evolving threats.

However, these IDS using deep learning methods introduce new vulnerabilities. For example, in this repository, we show how the use of a fast gradient sign method (FGSM) attack can help attackers understand how to craft their attacks in a way that would  "camouflage" their efforts from the IDS. We further show how the FGSM is less effective against ensemble models.



## Set-up
`git clone https://github.com/JonathanV1502/Defensive-ML.git`

### Dependences
``conda create -n DF-ML python=3.11``

``conda activate DF-ML``

``pip install -r requirements.txt``

### Loading Dataset
``python data.py``

``python preprocessing/cicids2017.py``

## Papers

1. [NAttack! Adversarial Attacks to bypass a GAN based classifier trained to detect Network intrusion](https://arxiv.org/pdf/2002.08527.pdf)

2. [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/pdf/1606.01583.pdf)

## Team
- Jonathan Villarreal
- Jaime Flores
- Alberto Castillo
- Axel Serrano

