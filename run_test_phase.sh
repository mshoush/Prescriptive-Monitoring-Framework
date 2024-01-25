#!/bin/bash

# Define an array of datasets
datasets=("bpic2012" "bpic2017")


# Function to process a dataset
process_dataset() {
    dataset="$1"
    echo "Start with $dataset..."

    # Set resource levels based on the dataset name
    if [ "$dataset" == "bpic2012" ]; then
	resources_list=(1 5 10 15)
	y=1
    elif [ "$dataset" == "bpic2017" ]; then
	y=2
	resources_list=(1 5 10 15)
    else
        echo "Unknown dataset: $dataset"
        return
    fi

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

    # Calculate the start and end CPU indexes for this dataset
    #start_cpu=0
    end_cpu=$((start_cpu + cpus_per_dataset - 1))
    echo "start_cpu: $start_cpu"
    echo "end_cpu: $end_cpu"

    # Iterate over dataset iterations
    for iteration in 1 2 3
    do
        echo "Iteration... $iteration"
        for resources in "${resources_list[@]}"
        do
            echo "Resources: $resources" 
	    taskset -c $start_cpu-$end_cpu  python run_testing_phase.py "$dataset" "results_all_runtime_v1" "$resources" "$iteration" ;
	    sleep 10s

        done
    done

    # Wait for all background processes to finish
    wait
}

# Iterate over datasets in the background
for d in "${datasets[@]}"
do
    process_dataset "$d" &
done

# Wait for all background processes to finish
wait


