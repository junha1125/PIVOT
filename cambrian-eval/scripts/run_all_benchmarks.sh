#!/bin/bash
set -e

echo "> run_all_benchmarks.sh $@"

ckpt="$1"
conv_mode="${2:-qwen_1_5}"

benchmarks=(
    mme
    mmbench_en
    mmbench_cn
    seed
    gqa

    scienceqa
    mmmu
    mathvista
    ai2d
    
    chartqa
    ocrbench
    textvqa
    # docvqa ## getting accuracy by submission is too time-consuming. so, we recommend utilzing lmms-eval code
    
    mmvp
    realworldqa
    coco
    ade
    omni

    # mmbench_cn
    # vizwiz
    # pope
    # mmvet
    # infovqa
    # stvqa
    # mmstar
    # synthdog
)

# Create a directory for checkpoint files if it doesn't exist
checkpoint_dir="checkpoints"
mkdir -p "$checkpoint_dir"

# Generate a unique checkpoint file name based on the script arguments
parts=(${ckpt//\// })
n=${#parts[@]}
if [[ "${parts[n-1]}" == checkpoint* ]]; then
    model_basename="${parts[n-3]}-${parts[n-2]}-${parts[n-1]}"
else
    model_basename="${parts[n-2]}-${parts[n-1]}"
fi

checkpoint_file="$checkpoint_dir/checkpoint_$model_basename.txt"
echo "Evaluation checkpoint txt path: $checkpoint_file"
script_dir=$(dirname $(realpath $0))

# Check if the checkpoint file exists and load the completed benchmarks
if [[ -f "$checkpoint_file" ]]; then
    completed_benchmarks=($(cat "$checkpoint_file"))
    echo "Resuming from checkpoint. Completed benchmarks: ${completed_benchmarks[@]}"
else
    completed_benchmarks=()
fi

start_time=$(date +%s)
for benchmark in "${benchmarks[@]}"; do
    if [[ " ${completed_benchmarks[@]} " =~ " $benchmark " ]]; then
        echo "Skipping completed benchmark: $benchmark"
        continue
    fi

    echo "Running benchmark: $benchmark"
    bash $script_dir/run_benchmark.sh --benchmark $benchmark --ckpt $ckpt --conv_mode $conv_mode
    echo "Finished benchmark: $benchmark"

    # Append the completed benchmark to the checkpoint file
    echo "$benchmark" >> "$checkpoint_file"

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    printf "Elapsed time: %02d:%02d \n\n" "$minutes" "$seconds"
done

total_end_time=$(date +%s)
total_elapsed=$((total_end_time - start_time))
total_minutes=$((total_elapsed / 60))
total_seconds=$((total_elapsed % 60))
printf "Finished all benchmarks\n"
printf "Total elapsed time: %02d:%02d (MM:SS)\n" "$total_minutes" "$total_seconds"