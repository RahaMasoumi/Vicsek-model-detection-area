# Vicsek-model-detection-area
The base code has been written by "Romain Simon" during summer 2021 in CPT lab, Marseille, France and I have developed it during winter 2023. This project aims to look at empirical contact networks from the simple particle models point of view. The quesion is: Are we able to reproduce some dynamical properties of empirical contact networks from relatively simple crowd models? In this respect one of the particle models that we have investiated is Vicsek model. The Vicsek model focuses on collective behavior emerging from local interactions among particles. Inspired by the movement of flocks of birds or schools of fish, this model assumes that particles align their velocities with those of their neighbors within a certain interaction radius, giving rise to coherent motion patterns on a macroscopic scale.
I have developed the code written by "Romain Simon". I have introduced the detection area concept for particles, which determines how particles come into contact. Particles only recognize their half front circle which is their detection area. Hence if two particles are in detection area of each other we consider it a contact. The code produces "tij" file which is the timeline array for all pair of particles. 
