import glob
import os

def getPathofTraj(idx):
    """
    idx: 
        list or int, From 1 to 36 (NOT 0-35)
    @ return:
        List of Ground Truth
        List of Control Signal
    """
    state_files_dir = '../camera_calibration/outputs/Jan19/cut_dat_info/'
    control_signal_files_dir = '../real_robot_data/Jan14_2020_Control_Signals/'
    state_files_list = sorted(glob.glob(os.path.join(state_files_dir, '*.txt')))
    control_signal_files_list = sorted(glob.glob(os.path.join(control_signal_files_dir, '*.csg')))

    if type(idx) == list:
        idx = [i-1 for i in idx]
        stds = []
        csgs = []
        for i in idx:
            stds.append(state_files_list[i])
            csgs.append(control_signal_files_list[i])
        return stds, csgs

    return state_files_list[idx-1], control_signal_files_list[idx-1]

if __name__ == '__main__':
    stds, csgs = getPathofTraj([1,3,5])

    state_files = stds
    control_signal_files = csgs
    print(stds)
    print(csgs)

# idx = list(range(len(control_signal_files_list)))
# for i, rbd, csg in zip(idx, state_files_list, control_signal_files_list):
#     rbd_basename = os.path.basename(rbd)
#     csg_basename = os.path.basename(csg)
#     print(i, rbd_basename, csg_basename)


# ith = 3
# state_files = [state_files_list[ith]]
# control_signal_files = [control_signal_files_list[ith]]