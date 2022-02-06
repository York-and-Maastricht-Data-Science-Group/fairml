#!/usr/bin/bash
java -cp fairml.jar org.eclipse.epsilon.fairml.generator.FairML $1
filename=$1
old=".flexmi"
new=".py"
echo "${filename/${old}/${new}}"
p2j -o "${filename/${old}/${new}}"