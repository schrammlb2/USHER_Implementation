CSGs=(1 7 9 11 21 25 27 33)
for t in ${CSGs[@]}; do
    python gt_sigmoid_neldermead.py $t > outputs/sigmoid/NM_$t.out
    # echo $t > outputs/sigmoid/NM_$t.out
done