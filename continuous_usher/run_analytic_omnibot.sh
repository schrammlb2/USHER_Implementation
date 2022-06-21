
RANGE=1000

k=8
offset=".01"
clip=100.0
args='--entropy-regularization=0.02 --gamma=.975 --n-batches=10' 
for env_name in 'OmnibotGridworld'
do
	logfile="logging/$env_name.txt"
	echo "" > $logfile
	N=1
	epochs=25 #25
	agent="train_HER_mod.py"
	command="mpirun -np $N python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=$k --ratio-clip=$clip"
	delta_command="mpirun -np $N python -u $delta_agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=$k --ratio-clip=$clip"
	echo "command=$command"
	noise=0.0
	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_her.txt"
	echo -e "" > $rec_file

	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_usher.txt"
	echo -e "" > $rec_file
	for i in 0 
	do
		( echo "running $env_name, $noise noise, 2-goal ratio with delta-probability";
		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio) 
		(echo "running $env_name, $noise noise, 1-goal"; 
		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) 	 		
	done
done