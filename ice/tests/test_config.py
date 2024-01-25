import pytest 
import os
import random
import shutil
from glob import glob
from ice.configs import Config
from inspect import getmembers, isclass
from pathlib import Path

from ice.fault_diagnosis import models as fd_models
from ice.anomaly_detection import models as ad_models
from ice.health_index_estimation import models as hi_models
from ice.remaining_useful_life_estimation import models as rul_models


module_map = {
    "fault_diagnosis" : getmembers(fd_models, isclass),
    "anomaly_detection" : getmembers(ad_models, isclass),
    "health_index_estimation" : getmembers(hi_models, isclass),
    "remaining_useful_life_estimation" : getmembers(rul_models, isclass)
}
configs_list = [y for x in os.walk("ice/configs") for y in glob(os.path.join(x[0], '*.yaml'))]

def find_model_class(cfg):
    task_name = cfg.TASK
    return next((model_class for class_name, model_class in module_map[task_name] if cfg.MODEL.CLASS_NAME == class_name), None)

def generate_full_keys(config, path=""):
    for key, value in config.items():
        if not isinstance(value, Config):
            yield path + key
        else:
            for full_key in generate_full_keys(value, path + key + '.'):
                yield full_key

@pytest.fixture(scope="session")
def configs_cache():
    """
    Fixture to create a session-level cache of configurations.
    Loads each configuration file from the 'ice/configs' directory 
        and stores it in a cache.
    """

    cache = {}
    for config_path in configs_list:
        try:
            cfg = Config(config_path)
        except:
            cfg = None

        cache[config_path] = cfg

    return cache

@pytest.mark.parametrize("config_path", configs_list)
def test_config_loading(config_path, configs_cache):
    """
    Test to ensure each configuration file in the 'ice/configs' directory
        is loaded without errors.
    Asserts that the configuration object is not None for each path.
    """

    assert configs_cache[config_path] is not None

@pytest.mark.parametrize("config_path", configs_list)
def test_config_location(config_path, configs_cache):
    """
    Test to check if the task name in the configuration file 
        matches its folder name.
    Asserts that the task name in the configuration aligns 
        with the folder structure.
    """

    task_name = configs_cache[config_path].TASK
    folder_name = Path(config_path).parts[-2]
    assert task_name == folder_name

@pytest.mark.parametrize("config_path", configs_list)
def test_class_exists(config_path, configs_cache):
    """
    Test to verify that for each configuration, 
        there is a corresponding model class in the ICE framework.
    Asserts that the model class exists for the given configuration task.
    """

    cfg = configs_cache[config_path]
    model_class = find_model_class(cfg)
    assert model_class is not None

@pytest.mark.parametrize("models_module", [(task, class_name) for task in module_map for class_name, model_class in module_map[task]])
def test_config_exists(models_module, configs_cache):
    """
    Test to ensure that for each model class in the ICE framework, 
        there is at least one corresponding configuration file.
    Asserts that a configuration exists for each class in the module map.
    """

    task, class_name = models_module

    res = None
    for cfg in list(configs_cache.values()):
        if cfg is not None and "TASK" in cfg and "MODEL" in cfg and "CLASS_NAME" in cfg.MODEL and cfg.TASK == task and cfg.MODEL.CLASS_NAME == class_name:
            res = cfg

    assert res is not None

@pytest.mark.parametrize("config_path", configs_list)
def test_missing_keys(config_path, configs_cache):
    """
    Test to check for any missing keys in the configuration files 
        that are required by the model classes.
    Asserts that all keys required by model are present in the configuration.
    """

    cfg = configs_cache[config_path]
    model_class = find_model_class(cfg)
    config_full_keys = list(generate_full_keys(cfg))
    model_full_keys = [".".join(keys_list) for keys_list in model_class._param_conf_map.values()]

    assert set(model_full_keys) - set(config_full_keys) == set()

@pytest.mark.parametrize("config_path", configs_list)
def test_extra_keys(config_path, configs_cache):
    """
    Test to verify that there are no extra keys in the configuration files
        that are not used by the model classes.
    Asserts that the configuration only contains keys 
        that are relevant to the model.
    """

    cfg = configs_cache[config_path]
    model_class = find_model_class(cfg)
    config_full_keys = list(generate_full_keys(cfg))
    model_full_keys = [".".join(keys_list) for keys_list in model_class._param_conf_map.values()] + ["MODEL.CLASS_NAME", "TASK"]

    assert set(config_full_keys) - set(model_full_keys) == set()

@pytest.mark.parametrize("config_path", configs_list)
def test_values_types(config_path, configs_cache):
    """
    Test to ensure that the data types of values in the configuration files
        match the expected types in the model classes.
    Asserts that the type of each configuration value 
        matches the annotation in the model class.
    """
    cfg = configs_cache[config_path]
    model_class = find_model_class(cfg)
    
    for param, dict_path in model_class._param_conf_map.items():
        ann_class = model_class.__init__.__annotations__[param]
        cfg_value = cfg.path_get(dict_path)
        assert ann_class == type(cfg_value)

@pytest.mark.parametrize("config_path", configs_list)
def test_model_load_config(config_path, configs_cache):
    """
    Test to check if model classes can be successfully instantiated
        using their corresponding configuration files.
    Asserts that the model class can be initialized from the configuration
        without errors.
    """

    cfg = configs_cache[config_path]
    model_class = find_model_class(cfg)
    model_class.from_config(cfg)
    assert True


@pytest.fixture
def artifacts_train_path(configs_cache):
    """
    Fixture to create temporary training directory with training artifacts 
        and remove them after test completion.
    Loads first valid config in the cache, creates corresponding model, 
        generates new experiment name and stores training artifacts.
    """
    cfg = next((config for config in configs_cache.values() if config is not None), None)
    if cfg is None:
        yield None 
        return     

    model_class = find_model_class(cfg)
    model = model_class.from_config(cfg)

    random_name = str(random.randint(1, 100000000))
    while os.path.exists(os.path.join(model._output_dir, random_name)):
        random_name = str(random.randint(1, 100000000))

    model.name = random_name
    model._store_atrifacts_train()
    yield model._training_path

    shutil.rmtree(os.path.join(model._output_dir, model.name))

def test_store_artifacts_train(configs_cache, artifacts_train_path):
    """
    Test to check if model saves all the training artifacts 
        in the predefined directory.
    Asserts if there is no valid configs or some of the artifacts does not exist.
    """

    if artifacts_train_path is None:
        pytest.fail("No vaild configs has been found")
    assert os.path.exists(artifacts_train_path)

    folders = os.listdir(artifacts_train_path)
    assert len(folders) == 1

    assert os.path.exists(os.path.join(artifacts_train_path, folders[0]))
    assert os.path.exists(os.path.join(artifacts_train_path, folders[0], 'config.yaml'))

@pytest.fixture
def artifacts_inference_path(configs_cache):
    """
    Fixture to create temporary inference directory with training artifacts 
        and remove them after test completion.
    Loads first valid config in the cache, creates corresponding model, 
        generates new experiment name and stores inference artifacts.
    """

    cfg = next((config for config in configs_cache.values() if config is not None), None)
    if cfg is None:
        yield None
        return 

    model_class = find_model_class(cfg)
    model = model_class.from_config(cfg)

    random_name = str(random.randint(1, 100000000))
    while os.path.exists(os.path.join(model._output_dir, random_name)):
        random_name = str(random.randint(1, 100000000))

    model.name = random_name
    model._store_atrifacts_inference({"test_metirc" : 0.5})
    yield model._inference_path

    shutil.rmtree(os.path.join(model._output_dir, model.name))

def test_store_artifacts_inference(configs_cache, artifacts_inference_path):
    """
    Test to check if model saves all the inference artifacts 
        in the predefined directory.
    Asserts if there is no valid configs or some of the artifacts does not exist.
    """
    assert os.path.exists(artifacts_inference_path)

    folders = os.listdir(artifacts_inference_path)
    assert len(folders) == 1

    assert os.path.exists(os.path.join(artifacts_inference_path, folders[0]))
    assert os.path.exists(os.path.join(artifacts_inference_path, folders[0], 'config.yaml'))
    assert os.path.exists(os.path.join(artifacts_inference_path, folders[0], 'metrics.json'))