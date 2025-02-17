# Trabalho final da matéria de Cibernética/Aprendizagem de Máquina/Redes Neurais

## Geração dos dados

```
bash collecting.sh
bash collecting-dynamic.sh
```

## Treinamento da rede

```
python DQNAgent.py
python DQNAgent-dynamic.py
```

## Execução da arquitetura

```
python AIAgent.py
```

## Pre requisitos

### Bison

bison (GNU Bison) 2.3

### Flex

flex 2.6.4

### Env Mac

```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Fast Forward Planner

```
./FF-v2.3/ff -f ./problem1.pddl -o ./domain.pddl -i 0
```