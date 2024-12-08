#!/bin/sh

# Check if the number of arguments is less than 4
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <log_name> <resFolder> <resource_levels>"
    echo "Example: $0 bpic2012 resultsRL '1 4 6 12'"
    exit 1
fi

log_name="$1"
resFolder="$2"
resource_levels="$3"
# ./run_variants_with_BLs_resource_constrained.sh trafficFines resultsRL "1 18 27 55"

# log_name="$1"
# resFolder="$2"
# mode="$3" # ["BL1", "BL2", "ours" ]
# resource_levels="$4"
#  log_name: ["bpic2012", "bpic2017", "trafficFines"]
# # 12: 1 4 6 12
# # 17: 1 3 4 8
# # traffic: 1 18 27 55
# ./run_variants_with_BLs_resource_constrained.sh bpic2017 resultsRL "1 3 4 8" 10 ???
# ./run_variants_with_BLs_resource_constrained.sh trafficFines resultsRL "1 18 27 55" 10

echo "Log: $log_name"

# Convert the resource levels string to an array
IFS=' ' read -r -a levels <<< "$resource_levels"
echo "Resource Levels: $resource_levels"

# Define modes
modes=("BL1" "BL2" "ours")

# Loop over modes
for mode in "${modes[@]}"
do
    # Map mode to corresponding value
    if [ "$mode" = "BL1" ]; then
        mapped_mode="zahra"
    elif [ "$mode" = "BL2" ]; then
        mapped_mode="metzger"
    elif [ "$mode" = "ours" ]; then
        mapped_mode="mahmoud"
    fi

    echo "Mode: $mode (Mapped mode: $mapped_mode)"

    # Loop over iterations
    for i in 1 2 3
    do
        # Loop over resource levels
        for j in "${levels[@]}"
        do
            echo "Iteration: $i"
            echo "Resource Level: $j"
            echo "Start first: $log_name $mode"
            echo "Start $mode..."
            STARTTIME=$(date +%s)
            echo "StartTime: $STARTTIME"
            python rl/ppo_temp_cost_reward_noPred.py "$mapped_mode" ./resultsICPMTest/"$log_name"/"$log_name"/"$resFolder"/all/"$mode$i"/"$mode$j" "$j" 1 5 25 10 60 "$log_name" all > out.txt &
            sleep 60

            # ENDTIME=$(date +%s)
            # echo "ENDTIME: $ENDTIME"
            # echo "It takes $((ENDTIME - STARTTIME)) seconds to complete this task..."
            #rm out.txt &
        done
    done
done
# Wait for all background processes to finish
wait