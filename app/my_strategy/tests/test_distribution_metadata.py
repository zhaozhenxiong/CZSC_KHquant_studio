"""The installed wheel has a stable entry point and no private runtime payload."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[3]


def test_distribution_contract_preserves_data_manager_and_resources():
    configuration = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert configuration['project']['scripts']['khquant'] == 'my_strategy.cli:main'
    assert configuration['project']['requires-python'] == '>=3.12,<3.13'
    packages = configuration['tool']['setuptools']['packages']
    assert 'my_strategy.data_manager' in packages
    assert 'my_strategy.data' not in packages
    assert 'my_strategy.tests' not in packages
    resources = configuration['tool']['setuptools']['package-data']['my_strategy']
    assert 'web_dashboard/static/**/*' in resources
    assert 'configs/cost/*.yaml' in resources


def test_enabled_tushare_provider_and_calendar_tools_have_locked_client_dependencies():
    lines = (ROOT / 'app/requirements-czsc-runtime.lock').read_text(encoding='utf-8').splitlines()
    requirements = {item.name: str(item.specifier) for line in lines if line.strip() and not line.startswith('#')
                    for item in [Requirement(line)]}
    assert requirements['tushare'] == '==1.4.29'
    assert requirements['bs4'] == '==0.0.2'
    assert requirements['simplejson'] == '==4.1.1'
    assert requirements['websocket-client'] == '==1.9.0'
