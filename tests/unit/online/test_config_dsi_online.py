from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.configs.experiment.simul.online import ConfigDSIOnline, SimulType


def test_from_offline_successful_conversion():
    # Create a ConfigDSI instance with specific values
    config_dsi = ConfigDSI(
        c=0.5, a=0.3, k=5, failure_cost=0.9, S=10, num_target_servers=4
    )

    # Convert to ConfigDSIOnline using the class method
    online_config = ConfigDSIOnline.from_offline(config_dsi)

    # Check all copied and computed fields
    assert online_config.c == config_dsi.c
    assert online_config.a == config_dsi.a
    assert online_config.k == config_dsi.k
    assert online_config.failure_cost == config_dsi.failure_cost
    assert online_config.S == config_dsi.S
    assert online_config.num_target_servers == config_dsi.num_target_servers

    # Check online specific configurations
    assert online_config.c_sub == config_dsi.c / 10
    assert online_config.failure_cost_sub == config_dsi.failure_cost / 10
    assert online_config.total_tokens == 100
    assert online_config.wait_for_pipe == 0.1
    assert online_config.simul_type == SimulType.DSI
    assert online_config.maximum_correct_tokens == 20


def test_from_offline_default_values():
    # Create a ConfigDSI instance with default values
    config_dsi = ConfigDSI()

    # Convert to ConfigDSIOnline using the class method
    online_config = ConfigDSIOnline.from_offline(config_dsi)

    # Check that the defaults for online specific fields are correct
    assert online_config.total_tokens == 100
    assert online_config.wait_for_pipe == 0.1
    assert online_config.simul_type == SimulType.DSI
    assert online_config.maximum_correct_tokens == 20
