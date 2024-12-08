#!/bin/bash

 # Define an array of datasets
 datasets=("bpic2012" "bpic2017" "trafficFines")

# Function to process a dataset
process_dataset() {
    dataset="$1"
    echo "Start with $dataset..."

    # Determine the maximum range based on the dataset
    case $dataset in
        "bpic2012")
            max_range=13
	    y=1
            ;;
        "bpic2017")
            max_range=10
	    y=2
            ;;
        "trafficFines")
            max_range=60
	    y=3
            ;;
        *)
            echo "Invalid value for $dataset"
            exit 1
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
    
    
    # Iterate over dataset iterations
    for i in 1 2 3
    do
        echo "iteration... $i"
        for ((k = 1; k <= max_range; k++))
        do
            echo "Resources: $k out of: $max_range"
            # Call the Python script with parameters for this dataset and iteration
            # mode: mahmoud, results_dir, $k: # resources, 1: tdur, 5: -ve outcome cost,
            # 25: intervention cost, 10: gain resources, 60: gain outcome, $dataset: log, and all: component.
            taskset -c $start_cpu-$end_cpu python rl/ppo_temp_cost_reward_noPred.py mahmoud ./resultsOnline/resourcesv2_$dataset/resultsResources_$i/mahmoud $k 1 5 25 10 60 $dataset all > out.txt
            rm out.txt
            sleep 10
        done
    done
}


y=0
# Iterate over datasets and run them in parallel with 8 CPUs each
for dataset in "${datasets[@]}"
do
	# Call the process_dataset function in the background for each dataset
	process_dataset "$dataset" &
done

# Wait for all background processes to finish before exiting
wait
