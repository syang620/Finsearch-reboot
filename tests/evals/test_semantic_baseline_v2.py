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


def ending():
    return {'captured_cases':['a','b'],'evaluation_errors':[],'control_violations':[],
            'controls_after':{'ac_power':True,'low_power_mode':0,'browser_process_count':0,'heavy_non_model_processes':[]},
            'model_identities_unchanged':True,'index_unchanged':True}


@pytest.mark.parametrize('change',[{'control_violations':[{'reason':'battery'}]},
    {'model_identities_unchanged':False},{'index_unchanged':False},{'model_verification_error':'unavailable'},
    {'index_verification_error':'unavailable'},{'runtime_cleanup_error':'failed'},
    {'evaluation_errors':['unscored']},{'captured_cases':['a']},{'captured_cases':['a','a']},
    {'controls_after':{'ac_power':False}}])
def test_completed_capture_with_failed_controls_or_integrity_is_diagnostic_only(change):
    record={**ending(),**change}
    assert not runner.finalize_validity(record,['a','b'],['a','b'])
    assert record['status']=='invalid_diagnostic' and not record['official_baseline_eligible']
    assert record['invalidity_reasons']


def test_controlled_verified_complete_capture_can_publish_summary():
    record=ending()
    assert runner.finalize_validity(record,['a','b'],['a','b'])
    assert record['status']=='complete' and record['official_baseline_eligible']


def test_missing_identity_verification_is_not_assumed_success():
    record=ending(); record.pop('model_identities_unchanged')
    assert not runner.finalize_validity(record,['a','b'],['a','b'])


def test_incomplete_evaluation_cannot_publish_official_summary():
    assert not runner.finalize_validity(ending(),['a','b'],['a'])
