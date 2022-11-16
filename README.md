# Hazard Reading and Fitting

A tool incorporated within IPBSD to perform second-order hazard fitting if necessary.

Improved SAC/FEMA-compatible hazard fitting.

Compatible with intensity measures: Sa(T1) and Sa_avg

### Requirements

* Clone openquake.engine from https://github.com/gem/oq-engine, and open the terminal inside the clone directory. Activate the environment you want to install the package. Then, install the package via:
    
      pip install -e .
      pip install -r requirements.txt


### Modules
* Response Spectrum Generation

        Generates response spectrum from ground motion

* Hazard.psha

        Performs probabilistic seismic hazard assessment (PSHA) using the Openquake engine

* Hazard.spectrum

        Generates response spectrum based on specified return period and hazard curves
        
* Hazard.hazard

        Fitting of hazard function using the improved procedures

* RecordSelection

        Ground motion record selection using EZGM


### Todos
* [ ] Create a schema for the input hazard file format
* [ ] Create documentation for each funcion
