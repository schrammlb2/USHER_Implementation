N=6
RANGE=1000

env_name="Asteroids"
# agent="train_HER.py"
agent="train_HER_mod.py"
delta_agent="train_delta_agent.py"
k=8
offset=".02"
clip=10.0
args='--entropy-regularization=0.001 --n-test-rollouts=50 --n-cycles=500 --n-batches=4 --batch-size=1000'

pushd
cd ..
for env_name in "TorusFreeze4"  
do
	logfile="logging/$env_name.txt"
	echo "" > $logfile
	epochs=10
	gamma=.98
	alt_gamma=.98
	command="mpirun -np 1 python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset  --gamma=$gamma --replay-k=$k --ratio-clip=$clip"
	delta_command="mpirun -np 1 python -u $delta_agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset  --gamma=$gamma --replay-k=$k --ratio-clip=$clip"
	echo "command=$command"
	noise=0.0
	for pos_reward in true false 
	do
		base_command="mpirun -np 1 python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset  --gamma=$gamma --replay-k=$k --ratio-clip=$clip  --non-terminal-goals"
		if [[ $pos_reward == true ]]; then
			command="$base_command --positive-reward"
			reward_type="positive_reward"
		fi		
		if [[ $pos_reward == false ]]; then
			command="$base_command"
			reward_type="negative_reward"
		fi
	
		echo "command=$command"

		echo -e "\nrunning $env_name, $reward_type, $noise noise, ARCHER" >> $logfile
		rec_file="logging/recordings/name_$env_name""_$reward_type""__noise_$noise""__agent_her.txt"
		echo saving to $rec_file
		echo -e "" > $rec_file
		for i in 0 1 2 3 4 5
		do 
			(echo "running $env_name, $reward_type, $noise noise, ARCHER"; 
			$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --archer) &
		done
		wait

		echo -e "\nrunning $env_name, $reward_type, $noise noise, 1-goal" >> $logfile
		rec_file="logging/recordings/name_$env_name""_$reward_type""__noise_$noise""__agent_usher.txt"
		echo saving to $rec_file
		echo -e "" > $rec_file
		for i in 0 1 2 3 4 5
		do 
			( echo "running $env_name, $reward_type, $noise noise, 2-goal ratio with delta-probability";
			$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio --delta-agent) &
		done
		wait
		
		echo -e "\nrunning $env_name, $reward_type, $noise noise, 1-goal" >> $logfile
		rec_file="logging/recordings/name_$env_name""_$reward_type""__noise_$noise""__agent_her.txt"
		echo saving to $rec_file
		echo -e "" > $rec_file
		for i in 0 1 2 3 4 5
		do 
			(echo "running $env_name, $reward_type, $noise noise, 1-goal"; 
			$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) &
		done
		wait
	done
done

popd