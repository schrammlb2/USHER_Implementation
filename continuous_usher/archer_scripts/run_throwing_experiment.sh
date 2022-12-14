
RANGE=1000

# offset=".01"
k=8
offset=".01"
clip=100.0
args='--entropy-regularization=0.001 --gamma=.9 --n-batches=100' # --batch-size=1000 --n-batches=10'
for env_name in 'Throwing'
do
	logfile="logging/$env_name.txt"
	# echo "" > $logfile
	N=7
	epochs=100
	agent="train_HER.py"
	command="mpirun -np $N python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=$k --ratio-clip=$clip"
	delta_command="mpirun -np $N python -u $delta_agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=$k --ratio-clip=$clip"
	echo "command=$command"
	for noise in 0.0
	do
		echo -e "\nrunning $env_name, $noise noise, 2-goal ratio with delta-probability" >> $logfile
		rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_usher.txt"
		echo saving to $rec_file
		echo -e "" > $rec_file
		for i in 0 1 2 3 4
		do
			( echo "running $env_name, $noise noise, 2-goal ratio with delta-probability";
			$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio) # --delta-agent)
		done

		echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
		rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_her.txt"
		echo saving to $rec_file
		echo -e "" > $rec_file
		for i in 0 1 2 3 4
		do
			(echo "running $env_name, $noise noise, 1-goal";
			$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) 
		done
	done
done
