import json


class ConfigNamespace:
    """
    Recursively converts a dictionary into an object with attribute-style access.
    Also provides a safe .get() method similar to dictionaries.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursive call for nested dictionaries
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Safe attribute access, returning default if key is missing."""
        return getattr(self, key, default)


class ReachRegConfig(ConfigNamespace):
    """
    Configuration object for the Reach-Reg method.
    Encapsulates river metadata, algorithm hyperparameters, and data provider settings.
    Inherits from ConfigNamespace to ensure all JSON fields are accessible.
    """

    def __init__(self, config_dict):
        # Initialize the base namespace with the full config dictionary
        super().__init__(config_dict)

        # Explicitly map critical river metadata for easier access and validation
        # These will look for attributes created by the super() call
        river_meta = config_dict.get('river_metadata', {})
        model_configs = config_dict.get('model_configs', {})

        # 1. River Metadata Mapping
        self.river_full_name = river_meta.get('river', self.river_name)
        self.up_reach = river_meta.get('up_reach')
        self.dn_reach = river_meta.get('dn_reach')
        self.country = river_meta.get('country', 'unknown').lower()
        self.metrical_crs = river_meta.get('metrical_crs', '3857')
        self.vs_with_neight_dams = river_meta.get('vs_with_neight_dams', [])
        self.sword_river_file = river_meta.get('sword_river_file')
        self.river_tributary_reaches = river_meta.get('river_tributary_reaches', [])
        self.gauge_dist_threshold = river_meta.get('gauge_dist_threshold', 10)

        # Ensure insitu_query_name is directly accessible even if nested in JSON
        self.insitu_query_name = river_meta.get('insitu_query_name', self.river_name)

        # 2. Algorithm Hyperparameters Mapping
        self.amp_thres = model_configs.get('amp_thres', 1)
        self.rmse_thres = model_configs.get('rmse_thres', 10)
        self.single_rmse_thres = model_configs.get('single_rmse_thres', 0.2)
        self.itpd_method = model_configs.get('itpd_method', 'akima')
        self.buffer = model_configs.get('buffer', 300)
        self.corr_thres = model_configs.get('corr_thres', 0.75)
        self.bottom = model_configs.get('bottom', 0.1)

        # 3. Data Provider Config (DAHITI)
        # Handle different formats of dahiti_collections from JSON
        raw_collections = config_dict.get('dahiti_collections')
        if isinstance(raw_collections, list):
            self.dahiti_collections = raw_collections
        elif isinstance(raw_collections, dict):
            self.dahiti_collections = raw_collections.get(self.country, [])
        else:
            self.dahiti_collections = []

    def __repr__(self):
        return f"<ReachRegConfig for {self.river_name} (Country: {self.country})>"