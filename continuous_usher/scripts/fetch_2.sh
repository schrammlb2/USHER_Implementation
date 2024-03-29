
RANGE=1000

# offset=".01"
k=8
offset=".01"
clip=.3
args='--entropy-regularization=0.001' # --batch-size=1000 --n-batches=10'

for env_name in 'FetchSlide-v1' #'FetchPush-v1' 
do
	logfile="logging/$env_name.txt"
	# echo "" > $logfile
	N=6
	epochs=5
	if [[ $env_name == 'FetchReach-v1' ]]; then
		epochs=5
		N=1
	fi
	if [[ $env_name == 'FetchPush-v1' ]]; then
		epochs=40
	fi
	if [[ $env_name == 'FetchSlide-v1' ]]; then
		epochs=150
	fi
	noise=0.0
	agent="train_HER.py"
	delta_agent="train_delta_agent_high_dim.py"
	command="mpirun -np $N python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=$k --ratio-clip=$clip"
	delta_command="mpirun -np $N python -u $delta_agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset --replay-k=0 --ratio-clip=$clip"
	echo "command=$command"	
	# if [[ $env_name == 'FetchPush-v1' ]]; then	
	# 	echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
	# 	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_her.txt"
	# 	echo saving to $rec_file
	# 	echo -e "" > $rec_file
	# 	for i in 0 1 2 #5 6
	# 	do 
	# 		(echo "running $env_name, $noise noise, 1-goal, run #$i"; 
	# 		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) 
	# 	done
	# 	echo -e "\nrunning $env_name, $noise noise, 2-goal ratio with delta-probability" >> $logfile
	# 	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_usher.txt"
	# 	echo saving to $rec_file
	# 	echo -e "" > $rec_file
	# 	for i in 0 1 2 # 5 6 
	# 	do 
	# 		( echo "running $env_name, $noise noise, 2-goal ratio with delta-probability, run #$i";
	# 		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio) 
	# 	done
	# fi
	echo -e "\nrunning $env_name, $noise noise, DDPG" >> $logfile
	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_DDPG.txt"
	echo saving to $rec_file
	echo -e "" > $rec_file
	for i in 0 1 2 #5 6
	do 
		(echo "running $env_name, $noise noise, DDPG, run #$i"; 
		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))  --replay-k=0) 
	done
	echo -e "\nrunning $env_name, $noise noise, δ-DDPG" >> $logfile
	rec_file="logging/recordings/name_$env_name""__noise_$noise""__agent_delta_ddpg.txt"
	echo saving to $rec_file
	echo -e "" > $rec_file
	for i in 0 1 2  #5 6 
	do 
		( echo "running $env_name, $noise noise, δ-DDPG, run #$i";
		$delta_command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) 
	done
done