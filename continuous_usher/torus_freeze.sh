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
	for noise in 0 
	do 
		# echo -e "\nrunning $env_name, $noise noise, 2-goal ratio with delta-probability" >> $logfile
		# for i in 0 1 2 3 4 
		# do 
		# 	( echo "running $env_name, $noise noise, 2-goal ratio with delta-probability";
		# 	$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio) &
		# done
		# wait
		# echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
		# for i in 0 1 2 3 4 5 6
		# do 
		# 	(echo "running $env_name, $noise noise, 1-goal"; 
		# 	$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) &
		# done
		# wait
		# echo -e "\nrunning $env_name, $noise noise, Q-learning" >> $logfile
		# for i in 0 1 2 3 4 5 6
		# do 
		# 	(echo "running $env_name, $noise noise, Q-learning"; 
		# 	$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --replay-k=0) &
		# done
		# wait
		echo -e "\nrunning $env_name, $noise noise, δ-DDPG" >> $logfile
		for i in 0 1 2 3 4 5 6 
		do 
			( echo "running $env_name, $noise noise, δ-DDPG";
			$delta_command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) &
		done
		wait
	done
done