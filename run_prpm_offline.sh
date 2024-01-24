#!/bin/bash

# List of datasets to process
datasets=("bpic2017"  "bpic2012")
#datasets=("bpic2017")


# Define a function to process each dataset
process_dataset() {
    dataset="$1"
    y=1  # Initialize y to 0

    case $dataset in
        "bpic2012")
            y=1
            ;;
        "bpic2017")
            y=2
            ;;
        "trafficFines")
            y=3
            ;;
    esac

    # Get the number of available CPUs
    num_cpus=$(nproc --all)

    # Calculate the number of CPUs to allocate for this dataset (up to a maximum of 8)
    cpus_per_dataset=$((num_cpus / ${#datasets[@]}))  # Divide by the number of datasets
    
    if [ "$cpus_per_dataset" -gt 8 ]; then
            cpus_per_dataset=8
    elif [ "$cpus_per_dataset" -gt 4 ]; then
            cpus_per_dataset=4
    elif [ "$cpus_per_dataset" -gt 2 ]; then
            cpus_per_dataset=2
    fi
    
    
    # Calculate the start and end CPU indexes for this dataset
    if [ "$y" -eq 1 ]; then
            start_cpu=0
    else
            x=$((y - 1))
            start_cpu=$((cpus_per_dataset * x))
    fi

    end_cpu=$((start_cpu + cpus_per_dataset - 1))
    echo "start_cpu: $start_cpu"
    echo "end_cpu: $end_cpu"

     # Use taskset to specify CPU affinity for each dataset
    # start preedictive model
    echo "Start preedictive model: $dataset"
    taskset -c $start_cpu-$end_cpu python -W ignore predictive_model/get_catboost_pred_uncer3.py "$dataset" "results/predictive/$dataset/" 50 > "out_preds_$dataset.txt"
    sleep 5

    # start conformal predictive model
    echo "Start conformal predictive model: $dataset"
    taskset -c $start_cpu-$end_cpu python -W ignore conformal_prediction/cp.py "$dataset" "./results/predictive/$dataset" "./results/conformal/$dataset" "./results/causal/$dataset" > "out_conformal_preds_$dataset.txt"
    sleep 5

    # start causal model
    echo "start causalLift model: $dataset"
    taskset -c $start_cpu-$end_cpu python3.6 -W ignore causal/causalLift_mutm.py "$dataset" > "out_causalLift_$dataset.txt"

    # start survival model
    echo "start survival model: $datase"
    taskset -c $start_cpu-$end_cpu python -W ignore survival_model/survival_model.py "$dataset" > "out_survival_$dataset.txt"
    sleep 5


    # start conformal causal model
    echo "start conformal causal model: $datase"
    taskset -c $start_cpu-$end_cpu Rscript conformal_prediction/causal_cp4.r "$dataset" > "out_conformal_causal_$dataset.txt"
    sleep 5


    echo "Processing completed for $dataset."
}

# Iterate over datasets and run them in parallel
for dataset in "${datasets[@]}"
do
    # Call the process_dataset function in the background for each dataset
    process_dataset "$dataset" &
done

# Wait for all background processes to finish before exiting
wait
