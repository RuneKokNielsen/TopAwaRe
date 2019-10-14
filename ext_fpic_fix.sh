#!/bin/bash

# External nifti library lacks the -fPIC compiler flag needed for this build.
# As the makefile is not dynamically configurable, this is a hack to add the flag.

echo "ADD -fPIC TO NIFTILIB MAKEFILE"

ln=$(grep -n -m 1 GNU_ANSI_FLAGS ext/niftilib/Makefile | cut -f1 -d:)


line=$(grep -m 1 GNU_ANSI_FLAGS ext/niftilib/Makefile)
line2="${line} -fPIC"

nlines=$(wc -l ext/niftilib/Makefile | cut -f1 -d " ")

ln1="$((nlines - ln))"

pre=$(head -n $ln ext/niftilib/Makefile)
post=$(tail -n $ln1 ext/niftilib/Makefile)

printf "${pre} -fPIC\n${post}" > ext/niftilib/Makefile

