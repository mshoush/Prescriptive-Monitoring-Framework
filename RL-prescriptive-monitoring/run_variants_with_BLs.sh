#!/bin/sh

# Check if the number of arguments is less than 3
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <log_name> <resFolder> <mode>"
    echo "Example: $0 bpic2012 mahmoud BL1"
    exit 1
fi

log_name="$1"
resFolder="$2"
mode="$3"

echo "Log: $log_name, mode: $mode"

if [ "$mode" != "BL1" ] && [ "$mode" != "BL2" ]; then
    # Resource levels are provided
    if [ "$#" -lt 4 ]; then
        echo "Usage: $0 <log_name> <resFolder> <mode> <resource_levels>"
        echo "Example: $0 bpic2012 mahmoud mahmoud '1 4 6 12'"
        exit 1
    fi
    resource_levels="$4"
    echo "Resource Levels: $resource_levels"

    # Convert the resource levels string to an array
    IFS=' ' read -r -a levels <<< "$resource_levels"
fi

if [ "$mode" != "BL1" ] && [ "$mode" != "BL2" ]; then
    # Mode is not "BL1" or "BL2," so we vary the components
    components=("all" "withoutTU" "withoutCIW" "withCATE")
else
    # Mode is "BL1" or "BL2," so set components to "all"
    components=("all")
fi

for i in 1 2 3
do
    if [ "$mode" != "BL1" ] && [ "$mode" != "BL2" ]; then
        for j in "${levels[@]}"
        do
            for m in "${components[@]}"
            do
                # Rest of your script remains unchanged
                echo "Component...$m"
                echo "Iter....$i"
                echo "Start first: $log_name $mode" ;
                echo "Start $mode..." ;
                STARTTIME=$(date +%s)
                echo "StartTime: $STARTTIME"            
                python rl/ppo_temp_cost_reward_noPred.py "$mode" ./resultsICPMTest/"$log_name"/"$log_name"/"$resFolder"/"$m"/"$mode$i"/"$mode$j" "$j" 1 5 25 10 60 "$log_name" "$m" > out.txt;
                ENDTIME=$(date +%s)
                echo "ENDTIME: $ENDTIME"
                echo "It takes $((ENDTIME - STARTTIME)) seconds to complete this task..."
                rm out.txt&
            done
        done
    else
        # Handle the case when mode is "BL1" or "BL2" here, e.g., set mode to "zahra" or "metzger" and j to 1
        if [ "$mode" = "BL1" ]; then
            mode="zahra"
        elif [ "$mode" = "BL2" ]; then
            mode="metzger"
        fi
        j=1  # Set j to 1
        echo "Component...all"
        echo "Iter....$i"
        echo "Start first: $log_name $mode" ;
        echo "Start $mode..." ;
        STARTTIME=$(date +%s)
        echo "StartTime: $STARTTIME"            
        python rl/ppo_temp_cost_reward_noPred.py "$mode" ./resultsICPMTest/"$log_name"/"$log_name"/"$resFolder"/all/"$mode$i"/"$mode" "$j" 1 5 25 10 60 "$log_name" all > out.txt;
        ENDTIME=$(date +%s)
        echo "ENDTIME: $ENDTIME"
        echo "It takes $((ENDTIME - STARTTIME)) seconds to complete this task..."
        rm out.txt&
    fi
done




