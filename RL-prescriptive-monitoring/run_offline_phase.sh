#!/bin/bash

# List of datasets to process
datasets=("trafficFines" "bpic2017" "bpic2012")

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
    taskset -c $start_cpu-$end_cpu python -W ignore predictive_model/get_catboost_pred_uncer.py "$dataset" "results/predictive/$dataset/" 1 > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu python -W ignore causal/causallift_adapted.py "$dataset" "./results/causal/$dataset/" "./results/predictive/$dataset/" > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu python -W ignore causal/lower_upper_cate.py --data="$dataset" --results_dir="./results/causal/$dataset/" > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu Rscript test_cfcuasal.r "$dataset" > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu python -W ignore conformal_prediction/conformal_prediction.py "$dataset" "./results/predictive/$dataset/" "./results/conformal/$dataset/" "./results/causal/$dataset/" > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu python -W ignore conformalized_survival_model_final_v1.py "$dataset" > "out_$dataset.txt"
    sleep 5

    taskset -c $start_cpu-$end_cpu python -W ignore prepare_data_for_RL_V2.py "$dataset" > "out_$dataset.txt"

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


