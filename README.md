# Trabalho final da matéria de Cibernética/Aprendizagem de Máquina/Redes Neurais

## Pre requisitos

### Bison

bison (GNU Bison) 2.3

### Flex

flex 2.6.4

### gcc

Apple clang version 15.0.0 (clang-1500.3.9.4)
Target: arm64-apple-darwin23.6.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin


### Env Mac

python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

## Fast Forward Planner

./FF-v2.3/ff -f ./problem1.pddl -o ./domain.pddl -i 0

## Dataset

bash collecting.sh

## Trainamento

python train.py

