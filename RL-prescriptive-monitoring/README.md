# Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach

This project contains supplementary material for the article ["Prescriptive Process Monitoring Under Resource Constraints: A Reinforcement Learning Approach"](https://arxiv.org/abs/2307.06564) by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en) and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring approach that relies on reinforcement Learning and conformal prediction to construct an intervention policy under limited resources. 

This paper investigated the hypothesis that the incorporation of significance, urgency, and capacity factors can augment the process of training an RL agent for triggering interventions in the context of a prescriptive process monitoring system. The paper specifically investigated this question in the context where there is a limited number of resources available to perform interventions in the process.



# Dataset: 
Original and prepared datasets can be downloaded from the following link:
* [BPIC2017, BPIC2012, and traficFines, i.e., a loan application and road fines processes](https://owncloud.ut.ee/owncloud/s/5zpcwR8rtpMC7Ko)



# Reproduce results:
To reproduce the results, please run the following:

* First, install the required packages using the following command into your environment:

                                  pip install -r requirementsRL.txt

* Next, download the data folder from the abovementioned link

* Run the following notebooks to prepare the datasets:
  
                                  prepare_trafficFines.ipynb
                                  prepare_data_bpic2012.ipynb
                                  prepare_data_bpic2017.ipynb

  
*   Run the following shell script to start experiments w.r.t the offline phase: 

                                     ./run_offline_phase.sh
    
*   Execute the following shell script to initiate experiments with varying resource availability, thereby obtaining resource utilization levels:

                                     ./run_extract_utilization_levels.sh

    
*   Compile results to extract the resource utilization levels by executing the following notebook:

                                     extract_resource_utilization_levels.ipynb


*   Run the following shell script to conduct experiments involving different variants of the proposed approach as well as baseline methods:

                                    ./run_variants_with_BLs.sh <log_name> <resFolder> <mode> <resource_levels>
                                    log_name: ["bpic2012", "bpic2017", "trafficFines"]
                                    mode: ["BL1", "BL2", "ours" ]
                                    resource_levels: as extracted from the previous step.
    
                                    Ex: taskset -c 0-7 ./run_variants_with_BLs.sh bpic2012  resultsRL ours  "1 4 6 12"
 
                                     

* Finally, execute the following notebook to collect results regarding RQ1 and RQ2.: 

                                     compile_results.ipynb
                                     




