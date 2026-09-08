import pytest
from scripts.evals.agents import run_semantic_baseline_v2 as runner


@pytest.mark.parametrize('change',[{'ac_power':False},{'low_power_mode':1},{'browser_process_count':1},
                                  {'heavy_non_model_processes':[{'process':'work','cpu':80}]}])
def test_controlled_baseline_refuses_uncontrolled_start(change):
    state={'ac_power':True,'low_power_mode':0,'browser_process_count':0,'heavy_non_model_processes':[]}
    with pytest.raises(ValueError): runner.check_controls({**state,**change})


def test_controlled_start_without_changing_system_settings():
    runner.check_controls({'ac_power':True,'low_power_mode':0,'browser_process_count':0,'heavy_non_model_processes':[]})


def test_baseline_stays_blocked_before_optimization_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'DATA',tmp_path)
    monkeypatch.setattr(runner,'clean_checkout',lambda:None)
    monkeypatch.setattr(runner,'committed_approval',lambda _: {'status':'approved_for_narrow_semantic_v2_baseline'})
    with pytest.raises(ValueError,match='freeze/approval'):
        runner.verify_freeze(tmp_path/'approval.json')
