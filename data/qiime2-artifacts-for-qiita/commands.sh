trim_lengths="90 100 150"

for tl in `echo $trim_lengths`; do
  for s in `grep -v '>' ../newbloom.all.fna | cut -c 1-${tl} | sort | uniq`; do
    echo -e ">$s\n$s" >> blooms-${tl}.fna
  done ;
done

for fna in `ls *.fna`; do
  qiime tools import \
    --input-path ${fna} \
    --output-path ${fna/fna/qza} \
    --type FeatureData[Sequence]
done
