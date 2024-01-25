import yaml
from typing import List

class Config(dict):

    def __init__(self, cfg_path= None):
        super(Config, self).__init__()

        if cfg_path is not None:
            with open(cfg_path, 'r') as f:
                new_config_dict = yaml.safe_load(f)

            Config._update_from_dict(self, new_config_dict)

    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __delattr__(self, name):
        super(Config, self).__delattr__(name)
        super(Config, self).__delitem__(name)

    __delitem__ = __delattr__

    def to_yaml(self, path : str):
        """Dump config to a YAML file at the specified path.

        Args:
            path (str): A path for YAML file.
        """
        output_dict = self._to_dict()
        with open(path, 'w') as file:
            yaml.dump(output_dict, file, default_flow_style=False)
        
    def _to_dict(self):
        conf_dict = {}
        for key in self.keys():
            if isinstance(self[key], Config):
                conf_dict[key] = self[key]._to_dict()
            else:
                conf_dict[key] = self[key]

        return conf_dict 
    
    def path_set(self, path : List[str], value):
        """Set value for config's field defined with path. Create fields, if path doesn't exists.

        Args:
            path (List[str]): A path to the field in the config defined with sequence of nested fields.
            value (obejct): A value to be set for a specified field.
        """
        cur_conf = self
        for sub_attr_name in path[:-1]:
            if sub_attr_name not in cur_conf:
                cur_conf[sub_attr_name] = Config()

            cur_conf = cur_conf[sub_attr_name]

        cur_conf[path[-1]] = value

    def path_get(self, path: List[str]):
        """Get value from config's field defined with path.

        Args:
            path (List[str]): A path to the field in the config defined with sequence of nested fields.

        Returns:
            object: A value from specified field.
        """
        cur_conf = self
        for sub_attr_name in path[:-1]:
            cur_conf = cur_conf[sub_attr_name]

        return cur_conf[path[-1]]
       
    @staticmethod    
    def _update_from_dict(config, new_config_dict):  
        for key, val in new_config_dict.items():
            if not isinstance(val, dict):
                config[key] = val
                continue
            if key not in config:
                config[key] = Config()
            Config._update_from_dict(config[key], val)

        return config
