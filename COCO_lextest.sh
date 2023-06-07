#!/bin/bash


if [ "$#" -lt 2 ]; then
  echo "Not enough input arguments (requires audio and embedding paths)"
  exit 2
fi

if [ "$#" -eq 2 ]; then
  matlab -batch "COCO_lextest '$1' '$2';" > /dev/null;
elif [ "$#" -eq 3 ]; then
  matlab -batch "COCO_lextest '$1' '$2' '$3';" > /dev/null;
elif [ "$#" -eq 4 ]; then
  matlab -batch "COCO_lextest '$1' '$2' '$3' $4;" > /dev/null;
elif [ "$#" -eq 5 ]; then
  matlab -batch "COCO_lextest '$1' '$2' '$3' $4 '$5';" > /dev/null;
fi

#grep "recall" tmp.txt | { grep -v grep || true; } > output.txt;
#rm tmp.txt

echo "CDI_lextest finished."
