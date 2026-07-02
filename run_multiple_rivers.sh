#!/bin/bash

source Bekker_Python/bin/activate

# Lista Twoich plików JSON

rivers=(

  "config_po.json"

  "config_missouri.json"

  "config_mississippi.json"

  "config_odra.json"

  "config_ganges.json"

  "config_elbe.json"

  "config_rhine.json"

  "config_solimoes.json"

)



for config in "${rivers[@]}"

do

   echo "Starting: $config"

   python -E run.py "$config"

   echo "Finished: $config"

   echo "----------------------"

done
