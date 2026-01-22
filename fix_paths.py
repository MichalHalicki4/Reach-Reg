import json

import os



# Twoja nowa baza danych na Linuxie

linux_base = "/home/michal.halicki/Desktop/Bekker/SWORD_rivers/SWORD_v17b_shp/"



# Lista Twoich plików config

configs = [f for f in os.listdir('.') if f.startswith('config_') and f.endswith('.json')]



for c_file in configs:

    with open(c_file, 'r') as f:

        data = json.load(f)

    

    # Pobieramy nazwę pliku (np. eu_sword_reaches_hb23_v17b.shp)

    original_path = data['river_metadata']['sword_river_file']

    filename = os.path.basename(original_path)

    

    # Określamy region na podstawie nazwy pliku (EU, NA, SA, AS)

    region = filename[:2].upper()

    

    # Składamy nową ścieżkę

    new_path = os.path.join(linux_base, region, filename)

    

    data['river_metadata']['sword_river_file'] = new_path

    

    with open(c_file, 'w') as f:

        json.dump(data, f, indent=4)

    print(f"Updated {c_file} -> {new_path}")
