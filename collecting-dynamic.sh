#!/bin/bash

total_start=$(date +%s)  # Marca o tempo total no início

for i in {1..200}; do
    start=$(date +%s)  # Marca o tempo de início da execução individual
    python generate-dynamic.py "$i"
    end=$(date +%s)  # Marca o tempo de término da execução individual

    duration=$((end - start))  # Calcula a duração da execução individual
    echo "Execução $i demorou $duration segundos"
done

total_end=$(date +%s)  # Marca o tempo total no final
total_duration=$((total_end - total_start))

echo "Tempo total: $total_duration segundos"
