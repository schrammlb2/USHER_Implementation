# CSGs=(11 21) # 1 7 9 11 21 25
CSGs=(1 7 9 11 21 25 27 33)

## Don't forget to update mu fist...

for t in ${CSGs[@]}; do
    # python plt_comparison.py $t
    python plt_comp_savedf.py $t
    python plt_comp.py $t
    # echo $t > ./outputs/gt_LBFGS_CSG$t.out
done