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

                                  pip install -r requirements.txt

* Next, download the data folder from the abovementioned link

* Run the following notebooks to prepare the datasets:
  
                                  ./prepare_data/prepare_data.ipynb


  
*   Run the following shell script to start experiments w.r.t the training and calibration phases: 

                                     ./run_prpm_offline.sh


*  To execute realCause, please run the following two commands and change the <data_name> to either 'bpic12' or 'bpic17'. For more details, please refer to the following [link](https://github.com/bradyneal/realcause):

                                      python realCause/train_generator.py --data <data_name>  --dist "SigmoidFlow" --n_hidden_layers 128 --dim_h 14 --w_transform "Standardize" --y_transform "Normalize" --saveroot "results_realcause_bpic2012"
                                      python realCause/make_datasets.py --data <data_name>


* Then, execute the following notebook to collect results regarding the testing phase, i.e., run-time.: 

                                     compile_results.ipynb


* Now, run the following shell script to conduct experiments involving different variants of the proposed framework as well as baseline methods:

                                    ./run_test_phase.sh

* Finally, execute the following notebook to collect results: 

                                     plot_results_all.ipynb




