import numpy as np
import matplotlib.pyplot as plt
import pickle

pospath = 'no1/1_1pos'
angpath = 'no1/1_1ang'
truepath = '../real_data/real_data/traj1_1.txt'
with open(pospath, 'rb') as pickle_file:
    poses = pickle.load(pickle_file)
with open(angpath, 'rb') as pickle_file:
    loads = pickle.load(pickle_file)
ground_truth = np.loadtxt(truepath, usecols=(5,6,7,1,2,3,4))
poses = np.asarray(poses)
loads = np.asarray(loads)

pos_guesses = np.reshape(poses, (-1, 2))
pos_truths = ground_truth[:, :2]
pos_mse = ((pos_guesses - pos_truths)**2).mean(axis=None)
print("Position MSE: ", pos_mse)
load_guesses = np.reshape(loads, (-1, 1))
load_truths = ground_truth[:, 2:3]
load_mse = ((load_guesses - load_truths)**2).mean(axis=None)
print("Load MSE: ", load_mse)
state_guesses = np.hstack((pos_guesses, load_guesses))
state_truths = ground_truth[:, :3]
state_mse = ((state_guesses - state_truths)**2).mean(axis=None)
print("State MSE: ", state_mse)

plt.figure(1)
plt.scatter(ground_truth[0, 0], ground_truth[0, 1], marker="*", label='start')
plt.plot(ground_truth[:, 0], ground_truth[:, 1], color='blue', label='Ground Truth')
plt.plot(poses[:, 0], poses[:, 1], color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Trajectory 1_1 Prediction -- Pos')
plt.legend()
plt.figtext(0.5, 0.01, "Pos MSE: " + str(pos_mse), wrap=True, horizontalalignment='center', fontsize=12)
plt.show()

plt.figure(2)
plt.scatter(ground_truth[0, 2], 0, marker="*", label='start')
plt.plot(ground_truth[:, 2], np.zeros_like(ground_truth[:, 2]), color='blue', label='Ground Truth')
plt.plot(loads[:, 0], np.full_like(loads[:, 0], 100), color='red', label='NN Prediction')
plt.axis('scaled')
plt.title('Trajectory 1_1 Prediction -- Ang')
plt.legend()
plt.figtext(0.5, 0.01, "Ang MSE: " + str(load_mse), wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
