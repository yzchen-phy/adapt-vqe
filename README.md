
# ADAPT algorithms
Includes simulation code for performing ADAPT-VQE and ADAPT-QAOA calculations.


### Installation
1. Download
    
        git clone https://github.com/yzchen-phy/adapt-vqe.git
        cd adapt-vqe/

3. create virtual environment (optional)
         
        virtualenv -p python3 venv
        source venv/bin/activate

4. Install

        pip install .

5. run tests
    
        pytest test/*.py

### ADAPT-VQE

Method detailed in Nature Communications, 10, 3007, (2019):[article](https://www.nature.com/articles/s41467-019-10988-2?utm_source=other_website&utm_medium=display&utm_content=leaderboard&utm_campaign=JRCN_2_LW_X-moldailyfeed&error=cookies_not_supported&code=55be89d0-c3c3-4c68-aee9-7424ed20999f)
|
[preprint](https://arxiv.org/abs/1812.11173)

### ADAPT-QAOA 

Operator pool construction is specified in operator_pools_qaoa.py and the main functions are in qaoa_methods.py.
Examples of usage are included in the folder examples_adapt-qaoa. 

Method detailed in Physical Review Research 4, 033029 (2022):[article](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033029)
|
[preprint](https://arxiv.org/abs/2005.10258)

Simulation code used for the work "How Much Entanglement Do Quantum Optimization Algorithms Require?":[preprint](https://arxiv.org/abs/2205.12283) 
