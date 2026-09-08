"""Evaluation-only deterministic channels, not general natural-language truth.

Numeric grammar coverage and unknowns are explicit. Source-adjudicated support
and the independently validated secondary judge supply broader semantic checks.
"""
from decimal import Decimal, InvalidOperation
import ast
import hashlib
import math
import re

from evals.semantic_answer_v1 import evidence_text, rate, visible_contexts
from evals.semantic_numeric_v2 import assess_statement, assertion
from evals.semantic_outcomes_v2 import accepted_answer, execution


def decimal(value):
    if isinstance(value, bool) or value is None: return None
    try: value = Decimal(str(value).replace(',', ''))
    except InvalidOperation: return None
    return value if value.is_finite() else None


def canonical_prose(text):
    return re.sub(r'\s+', ' ', text.replace('**','').replace('__','').replace('’', "'")).strip()


def evidence_support(context, gold, catalog):
    """Exact proven source succeeds; absence from examples is unknown, not false.

    A valid ID alone cannot pass: the visible KB evidence must contain a frozen
    source-derived table representation. Prefix row evidence and presentation
    changes do not invalidate a table, but invisible/altered source rows cannot
    pass from a stable payload hash alone.
    """
    number = gold['numeric']
    if context.get('kind') == 'structured_fact':
        fact = context.get('structured_fact') or {}
        if fact.get('status') != 'ok': return 'unsupported'
        if any(fact.get(k) != number[k] for k in ('metric_id','ticker','fiscal_year','unit')): return 'unsupported'
        value = decimal(fact.get('value'))
        # Display tolerance applies to answer formatting, not to the raw source
        # fact. A nearby but different source value is not the annotated fact.
        if value is None or value != Decimal(str(number['value'])): return 'unsupported'
        # Exact source filing provenance is required for filing-scoped questions.
        # Runtime context that lacks it remains unknown, not assumed equivalent.
        originals = [s for s in gold['sources'] if s['kind'] == 'inline_xbrl']
        if not originals: return 'unknown'
        for source in originals:
            if any(fact.get(k) != source.get(k) for k in ('form_type','start_date','report_date')): continue
            # Filed-year guesses are forbidden; URL equality must be independently
            # source-adjudicated when original HTML provenance is unavailable.
            if fact.get('source_sha256') == source['source_sha256']: return 'supported'
        return 'unknown'
    if context.get('kind') not in {'text','table'}: return 'unsupported'
    source = context.get('source') or {}; payload = context.get('payload') or {}
    doc_id = source.get('doc_id') or payload.get('doc_id')
    visible=evidence_text(context)
    digest = hashlib.sha256(visible.encode()).hexdigest()
    links=catalog.get('facts',[]) if isinstance(catalog,dict) else catalog
    displays=catalog.get('displays',{}) if isinstance(catalog,dict) else {}
    candidates = [s for s in gold['sources'] if s['kind'] == 'kb']
    fact_ids = {s['fact_id'] for s in gold['sources'] if s['kind'] == 'inline_xbrl'}
    candidates += [r for r in links if r['fact_id'] in fact_ids]
    for candidate in candidates:
        if candidate['evidence_id'] != doc_id: continue
        display=displays.get(doc_id)
        if display is not None:
            if display['content_sha256']!=candidate['content_sha256']: continue
            for key in ('ticker','fiscal_year','form_type'):
                if any(container.get(key) is not None and container[key]!=display['metadata'][key] for container in (source,payload)):
                    return 'unsupported'
            if any(r['text'].strip() and r['text'].strip() in visible for r in display['representations']): return 'supported'
        if candidate['content_sha256'] == digest: return 'supported'
    return 'unknown'


def calculator_provenance(gold, claim, contexts, analyst, catalog):
    """Check selected computation, recorded call args and source-bound operands.

    The exported runtime trace records calls and selected result, not a separate
    raw tool-response ledger. This is reported trace provenance, not a claim of
    independently observing the calculator service's response.
    """
    if gold['claim_type'] != 'calculation': return {'status':'not_applicable'}
    computation = analyst.get('computation') or {}; trace = analyst.get('trace') or {}
    if trace.get('used_financial_evaluator') is not True: return {'status':'missing', 'reason':'calculator_not_recorded'}
    expression = computation.get('expression'); variables = computation.get('variables')
    value = decimal(computation.get('result'))
    if not isinstance(expression,str) or not isinstance(variables,dict) or value is None:
        return {'status':'unknown', 'reason':'selected_computation_incomplete'}
    calls = [c for c in trace.get('tool_calls',[]) if c.get('name') == 'financial_evaluator']
    normalize = lambda s: re.sub(r'\s+', '', s)
    def variables_equal(raw):
        return (isinstance(raw,dict) and set(raw)==set(variables)
                and all(decimal(raw[k]) is not None and decimal(raw[k])==decimal(variables[k]) for k in variables))
    if not any(isinstance(c.get('args'),dict) and normalize(str(c['args'].get('expression',''))) == normalize(expression)
               and variables_equal(c['args'].get('variables')) for c in calls):
        return {'status':'unknown', 'reason':'no_exact_recorded_call_binding'}
    refs = [contexts[c] for c in claim.get('context_ids',[]) if c in contexts]
    operands = []
    for source in gold['sources']:
        operand = {'claim_type':'structured_numeric', 'sources':[source], 'numeric':{
            'ticker':source['ticker'], 'metric_id':source['metric_id'], 'fiscal_year':source['fact_fiscal_year'],
            'value':source['value'], 'unit':source['unit'], 'absolute_tolerance':0}}
        if not any(evidence_support(c,operand,catalog)=='supported' for c in refs):
            return {'status':'unknown','reason':'operand_evidence_not_independently_bound'}
        operands.append(Decimal(str(source['value'])))
    # Bounded growth expression, not arbitrary eval. Variable names do not imply
    # roles; their numeric values must bind to the original source periods.
    try: tree=ast.parse(expression.strip(),mode='eval').body
    except (SyntaxError,ValueError): tree=None
    percent=False
    if isinstance(tree,ast.BinOp) and isinstance(tree.op,ast.Mult):
        if isinstance(tree.right,ast.Constant) and type(tree.right.value) in (int,float) and tree.right.value==100:
            tree=tree.left; percent=True
        elif isinstance(tree.left,ast.Constant) and type(tree.left.value) in (int,float) and tree.left.value==100:
            tree=tree.right; percent=True
    valid=(isinstance(tree,ast.BinOp) and isinstance(tree.op,ast.Div) and isinstance(tree.left,ast.BinOp)
           and isinstance(tree.left.op,ast.Sub) and isinstance(tree.left.left,ast.Name)
           and isinstance(tree.left.right,ast.Name) and isinstance(tree.right,ast.Name)
           and tree.left.right.id==tree.right.id)
    if not valid:
        return {'status':'unknown','reason':'expression_outside_bound_growth_grammar'}
    current = decimal(variables.get(tree.left.left.id)); previous = decimal(variables.get(tree.left.right.id))
    if current is None or previous is None or not previous:
        return {'status':'incorrect','reason':'invalid_bound_operands'}
    if not any(previous == operands[0]/scale and current == operands[1]/scale for scale in (Decimal(1),Decimal(10**6),Decimal(10**9))):
        return {'status':'incorrect','reason':'wrong_operand_entity_metric_period_or_scale'}
    calculated = (current-previous)/previous * (100 if percent else 1)
    if abs(calculated-value) > Decimal('0.00000001'):
        return {'status':'incorrect','reason':'selected_result_disagrees_with_expression'}
    percentage = value if percent else value*100
    if abs(percentage-Decimal(str(gold['numeric']['value']))) > Decimal(str(gold['numeric']['absolute_tolerance'])):
        return {'status':'incorrect','reason':'wrong_calculated_percentage'}
    return {'status':'supported','reason':'source_bound_operands_recorded_call_selected_result_agree',
            'limitation':'No independent raw calculator response is exported by this unchanged runtime.'}


def numeric_check(gold, output, contexts, catalog):
    answer = accepted_answer(output)
    if answer is None: return {'claim_id':gold['claim_id'], 'truth':'unassessed', 'evidence_support':'unassessed', 'credit':False}
    number = gold['numeric']; claims = answer.get('claims') or []
    candidates = []
    for claim in claims:
        parsed = assertion(claim['text'])
        if parsed['status'] == 'parsed' and all(parsed[k] == expected for k,expected in
                [('entity',number['ticker']),('metric_id',number['metric_id']),('fiscal_year',number['fiscal_year'])]):
            candidates.append(claim)
    # Wrong entity/period/metric cannot gain credit via declared metadata. This
    # fallback diagnoses a lone attempted numeric requirement, not a match.
    if not candidates and len(claims)==1: candidates=claims
    if not candidates: return {'claim_id':gold['claim_id'], 'truth':'missing' if not claims else 'unknown', 'evidence_support':'unknown', 'credit':False}
    results=[]
    answer_lines={canonical_prose(line) for line in answer['answer'].splitlines() if line.strip()}
    for claim in candidates:
        assessment = assess_statement(claim['text'],number)
        # No positive credit for claims retained outside the answer or embedded
        # under a negating prefix. Broader prose needs source semantic assessment.
        if canonical_prose(claim['text']) not in answer_lines:
            assessment={'status':'unknown','reason':'not_an_exact_standalone_answer_assertion'}
        refs=claim.get('context_ids',[])
        support=[evidence_support(contexts[c],gold,catalog) for c in refs if c in contexts] if gold['claim_type']!='calculation' else []
        evidence='supported' if 'supported' in support else ('unsupported' if not refs or any(c not in contexts for c in refs) or support and all(s=='unsupported' for s in support) else 'unknown')
        calc=calculator_provenance(gold,claim,contexts,output['analyst'],catalog)
        if gold['claim_type']=='calculation': evidence=calc['status'] if calc['status'] in {'supported','unsupported'} else 'unknown'
        if any(c not in contexts for c in refs): evidence='unsupported'
        results.append({'emitted_claim_id':claim['claim_id'], 'truth':assessment['status'], 'reason':assessment['reason'],
                        'evidence_support':evidence, 'calculator':calc,
                        'credit':assessment['status']=='correct' and evidence=='supported' and calc['status'] in {'supported','not_applicable'}})
    # Conflicting explicit assertions never resolve by selecting the lucky one.
    truths={r['truth'] for r in results}
    if 'incorrect' in truths: truth='incorrect'
    elif truths=={'correct'}: truth='correct'
    else: truth='unknown'
    evidence_statuses={r['evidence_support'] for r in results}
    combined_evidence='supported' if 'supported' in evidence_statuses else ('unsupported' if evidence_statuses=={'unsupported'} else 'unknown')
    return {'claim_id':gold['claim_id'], 'truth':truth, 'evidence_support':combined_evidence,
            'credit':truth=='correct' and any(r['credit'] for r in results), 'assertions':results}


def deterministic_case(case, output, catalog=()):
    outcome=execution(output); answer=accepted_answer(output)
    contexts=visible_contexts(output) if answer is not None else {}
    claims=(answer or {}).get('claims') or []
    covered=refs_total=refs_valid=structured_total=structured_ok=kb_total=kb_ok=0
    checks=[]
    for claim in claims:
        refs=set(claim.get('context_ids',[])); valid=refs & contexts.keys()
        covered+=bool(refs); refs_total+=len(refs); refs_valid+=len(valid)
        flags=[]
        if not refs: flags.append('missing_citation')
        if valid!=refs: flags.append('invalid_context_id')
        kind=claim.get('claim_type')
        if kind=='structured_numeric':
            structured_total+=1
            compatible=bool(claim.get('metric_id')) and any(contexts[c].get('kind')=='structured_fact' and (contexts[c].get('structured_fact') or {}).get('status')=='ok' and (contexts[c].get('structured_fact') or {}).get('metric_id')==claim.get('metric_id') for c in valid)
            structured_ok+=compatible
            if not compatible: flags.append('evidence_type_mismatch')
        elif kind in {'narrative','attribution','kb_numeric'}:
            kb_total+=1; compatible=any(contexts[c].get('kind') in {'text','table'} and evidence_text(contexts[c]).strip() for c in valid)
            kb_ok+=compatible
            if not compatible: flags.append('evidence_type_mismatch')
        checks.append({'claim_id':claim['claim_id'],'structural_flags':flags})
    return {'case_id':case['id'],'stratum':case['stratum'],'ticker':case['ticker'],'execution':outcome,
            'claim_checks':checks,'emitted_eligible_claims':len(claims),
            'claim_citation_coverage':rate(covered,len(claims)), 'valid_context_id_rate':rate(refs_valid,refs_total),
            'structured_evidence_compatibility':rate(structured_ok,structured_total), 'kb_evidence_compatibility':rate(kb_ok,kb_total),
            'numeric_checks':[numeric_check(g,output,contexts,catalog) for g in case['required_claims'] if g.get('numeric')],
            'structural_warning':'Citation validity/type compatibility are not semantic support.',
            'latency_ms':(output.get('orchestrator_trace') or {}).get('total_ms')}
