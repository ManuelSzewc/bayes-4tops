#!/bin/bash

echo "Bash version ${BASH_VERSION}"

# here is where you set every parameter you need to play with
file_dir="$1"
data_dir="$2"
N="$3"
nwalkers="$4"
T="$5"
burnout="$6"
keep_every="$7"


python3 make_data.py "$file_dir" "$data_dir"

sleep 10

for (( walker=1; walker<="$nwalkers"; walker++ )) #number of walkers
do
	echo "Walker $walker"
	#echo "$data_dir" "$data_dir"/walker_"$walker" "$N" "$T" "$burnout" "$keep_every"
	mkdir "$data_dir"/walker_"$walker" #make directory where I'll store everything
	python3 run_gibbs_sampler_one_walker.py "$data_dir" "$data_dir"/walker_"$walker" "$N" "$T" "$burnout" "$keep_every" # usage is data_dir output_dir Number of events considered Number of saved samples Burnin Space between samples
	sleep 60
done

python3 run_vb.py "$data_dir" "$N"

sleep 10

python3 run_plotter_all_walkers.py "$data_dir" "$nwalkers" "$N" "$T" "$burnout" "$keep_every"

python3 run_autocorrelation_and_thinning.py "$data_dir" "$nwalkers" "$N" "$T" "$burnout" "$keep_every"

mkdir "$data_dir"/thinned_plots

python3 run_plotter_all_thinned_walkers.py "$data_dir" "$nwalkers" "$N" "$T"

