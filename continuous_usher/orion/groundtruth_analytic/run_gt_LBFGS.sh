# CSGs=(11 21) # 1 7 9 11 21 25
CSGs=(1 7 9 11 21 25 27 33)

for t in ${CSGs[@]}; do
    # python gt_inference_raw_fixJacobian.py $t > ./outputs/gt_LBFGS_CSG$t.out
    python gt_derivative_based.py $t > ./outputs/LBFGS/gt_LBFGS_CSG$t.out
    # echo $t > ./outputs/gt_LBFGS_CSG$t.out
done