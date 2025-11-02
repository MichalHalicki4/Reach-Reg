all_river_data = {
    'Oder': {
        'river': 'Oder, River',
        'river_names': ['Oder, River'],
        'basin': 'Oder',
        'up_reach': 24222700041,
        'dn_reach': 24221000081,
        'country': 'poland',
        'metrical_crs': '2180',
        'vs_with_neight_dams': [41867, 23411],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp',
        'river_tributary_reaches': [24222100011],
        'gauge_dist_threshold': 5
    },
    'Elbe': {
        'river': 'Elbe, River',
        'river_names': ['Elbe, River'],
        'basin': 'Elbe',
        'up_reach': 23285000251,
        'dn_reach': 23281000101,
        'country': 'germany',
        'metrical_crs': '4839',
        'vs_with_neight_dams': [38476],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 5
    },
    'Rhine': {
        'river': 'Rhine, River',
        'river_names': ['Rhine, River'],
        'basin': 'Rhine',
        'up_reach': 23267000091,
        'dn_reach': 23261000151,
        'country': 'germany',
        'metrical_crs': '4839',
        'vs_with_neight_dams': [],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb23_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 5
    },
    'Po': {
        'river': 'Po, River',
        'river_names': ['Po, River'],
        'basin': 'Po',
        'up_reach': 21406901046,
        'dn_reach': 21406100345,
        'country': 'italy',
        'metrical_crs': '3035',
        'vs_with_neight_dams': [46234],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb21_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 5
    },
    'Missouri': {
        'river': 'Missouri, River',
        'river_names': ['Missouri, River'],
        'basin': 'Missouri',
        'up_reach': 74295500011,
        'dn_reach': 74291100011,
        'country': 'United States of America',
        'metrical_crs': 'ESRI:102010',
        'vs_with_neight_dams': [41781, 15919, 41780, 16074, 14684],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/NA/na_sword_reaches_hb74_v17b.shp',
        'river_tributary_reaches': [74295100011],
        'gauge_dist_threshold': 5
    },
    'Mississippi': {
        'river': 'Mississippi, River',
        'river_names': ['Mississippi, River'],
        'basin': 'Mississippi',
        'up_reach': 74270900041,
        'dn_reach': 74270100011,
        'country': 'United States of America',
        'metrical_crs': 'ESRI:102010',
        'vs_with_neight_dams': [22863, 36574, 46325, 36617, 46317],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/NA/na_sword_reaches_hb74_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 10
    },
    'Solimoes': {
        'river': 'Solimões, River',
        'river_names': ['Solimões, River'],
        'basin': 'Solimões',
        'up_reach': 62293900141,
        'dn_reach': 62293100061,
        'country': 'Brazil',
        'metrical_crs': 'ESRI:102033',
        'vs_with_neight_dams': [],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/SA/sa_sword_reaches_hb62_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 10
    },
    'Ganges': {
        'river': 'Ganges, River',
        'river_names': ['Ganges, River'],
        'basin': 'Ganges',
        'up_reach': 45243500161,
        'dn_reach': 45243100011,
        'country': 'India',
        'metrical_crs': 'ESRI:102025',
        'vs_with_neight_dams': [13062],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/AS/as_sword_reaches_hb45_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 10
    },
    'Wisła': {
        'river': 'Wisła, River',
        'river_names': ['Wisła, River', 'Vistula, River'],
        'basin': 'Wisla',
        'up_reach': 24244600846,
        'dn_reach': 24230000025,
        'country': 'poland',
        'metrical_crs': '2180',
        'vs_with_neight_dams': [],
        'sword_river_file':
            '/Users/michalhalicki/Documents/nauka/dane_gis/SWORD_v17b_shp/EU/eu_sword_reaches_hb24_v17b.shp',
        'river_tributary_reaches': [],
        'gauge_dist_threshold': 5
    }
}

dahiti_in_situ_collections = {
    'germany': [5, 44, 46, 48, 49, 50, 51, 53],
    # 'poland': [35]
    'poland': [57],
    'italy': [29]
}

configs = {
    'amp_thres': 1,
    'rmse_thres': 10,
    'single_rmse_thres': 0.2,
    'itpd_method': 'akima',
    'buffer': 300,
    'corr_thres': 0.75,
    'bottom': 0.1
}
