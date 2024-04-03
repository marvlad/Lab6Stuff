#!/bin/bash

# Input file
input_file="file.txt"

# Output file
output_file="new_file.txt"

# Extracting desired values using awk
awk '
/temperature/ {temperature = $3}
/humidity/ {humidity = $3}
/thermistor/ {thermistor = $3}
/V\(3\.3\)/ {v_3_3 = substr($5, 1, length($5)-2)}
/V\(3\.1\)/ {v_3_1 = substr($7, 1, length($7)-2)}
/V\(1\.8\)/ {v_1_8 = substr($9, 1, length($9)-2)}
/Threshold for DAC 0/ {threshold_dac_0 = $6}
/Threshold for DAC 1/ {threshold_dac_1 = $6}
/Saltbridge/ {saltbridge = $4}
END {
    print humidity, temperature, thermistor, v_3_3, v_3_1, v_1_8, threshold_dac_0, threshold_dac_1, saltbridge
}' "$input_file" >> "$output_file"

