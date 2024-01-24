# White Box Specification of Intervention Policies for Prescriptive Process Monitoring

This project contains supplementary material for the article ["White Box Specification of Intervention Policies for Prescriptive Process Monitoring"]() by [Mahmoud Shoush](https://scholar.google.com/citations?user=Jw4rBlkAAAAJ&hl=en) and [Marlon Dumas](https://kodu.ut.ee/~dumas/). We propose a prescriptive process monitoring framework that enables
analysts to define intervention policies in a white-box manner. The framework incorporates an automated method for tuning the parameters of the intervention policies
to optimize a total gain function. 

This framework allows analysts to specify their intervention policies via filtering and ranking policies. The framework takes into account the importance (need and effect) and urgency (time left to intervene) of the interventions that the PrPM system may trigger, as well as the uncertainty of an intervention decision and the available capacity to perform interventions. The framework additionally incorporates an automated parameter optimization method to help users find optimal tradeoffs between different intervention policy dimensions.



# Dataset: 
Original and prepared datasets can be downloaded from the following link:
* [BPIC2017, and BPIC2012, i.e., a loan origination processes](https://owncloud.ut.ee/owncloud/s/piyeP7sGHb3fdQ7)



# Reproduce results:
To reproduce the results, please run the following:

* First, install the required packages using the following command into your environment:

                                  pip install -r requirementsRL.txt

* Next, download the data folder from the abovementioned link

* Run the following notebooks to prepare the datasets:
  
                                  ./prepare_data/prepare_data.ipynb


  
*   Run the following shell script to start experiments w.r.t the training and calibration phases: 

                                     ./run_prpm_offline.sh


*  To execute realCause, please run the following two commands and change the data_name to either bpic12 or bpic17. For more details, please refer to the following [link]([https://kodu.ut.ee/~dumas/](https://github.com/bradyneal/realcause):

                                      python realCause/train_generator.py --data <data_name>  --dist "SigmoidFlow" --n_hidden_layers 128 --dim_h 14 --w_transform "Standardize" --y_transform "Normalize" --saveroot "results_realcause_bpic2012"
                                      python realCause/make_datasets.py --data <data_name>


    
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
                                     





