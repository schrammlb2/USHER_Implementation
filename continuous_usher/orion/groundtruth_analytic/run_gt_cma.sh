# CSGs=(11 21) # 1 7 9 11 21 25
CSGs=(1 7 9 11 21 25 27 33)

for t in ${CSGs[@]}; do
    # python gt_inference_raw_fixJacobian.py $t > ./outputs/gt_LBFGS_CSG$t.out
    python gt_cma.py $t >> ./outputs/CMAES/CMAES_CSG$t.out
done