# CSGs=(11 21) # 1 7 9 11 21 25
CSGs=(1 7 9 11 21 25 27 33)
# factor=(.3 1)
factor=(.1 .2 .3 .4 .5 .6 .7 .8 .9 1)

for t in ${CSGs[@]}; do
for f in ${factor[@]}; do
    python data_efficiency.py $t $f > ./outputs/DE/DE_CSG"$t"_f$f.out
done
done

