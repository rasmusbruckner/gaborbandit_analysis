# Gabor-Bandit Analyses

This repository contains the analysis code for the Gabor-Bandit project. For the corresponding task code, go to the [gaborbandit_task](https://github.com/rasmusbruckner/gaborbandit_task) repository.

## Getting Started

### Model-Based Analyses

For model-based analyses use the *gb_modelbased* script. This script runs the parameter estimation and the evaluation of the agent-based computational models. The script *gb_recovery* implements the parameter and model recovery studies. Use the *gb_postpred* script for posterior predictive checks to compare the agents to the participants. Results of these model-based analyses can be plotted with *gb_mbplot*.

To validate the analytical model code, we used *gb_sampval*. This script implements the sampling-based validation of the analytical results of agent A4.

For a demonstration of the agents, run *gb_demonstration*.

### Descriptive Analyses

For descriptive analyses use the *gb_descriptive* script. 

## Built With

* [Python](https://www.python.org)
* [NumPy](http://www.numpy.org) - Computations and data organization
* [SciPy](https://www.scipy.org) - Computations
* [pandas](https://pandas.pydata.org) - Data organization
* [matplotlib](https://matplotlib.org) - Plotting
* [seaborn](https://seaborn.pydata.org) - Plotting
* [tqdm](https://tqdm.github.io) - Progress bar 

## Authors

* **Rasmus Bruckner** - [GitHub](https://github.com/rasmusbruckner) - [IMPRS LIFE](https://www.imprs-life.mpg.de/de/people/rasmus-bruckner)
* **Dirk Ostwald** - [FU Berlin](http://www.ewi-psy.fu-berlin.de/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/people/ostwald/index.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

