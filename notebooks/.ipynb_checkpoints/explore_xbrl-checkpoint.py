import json

data = json.load(open('data/xbrl/AAPL_CIK0000320193_facts.json'))
print('Entity:', data['entityName'])
print('CIK:', data['cik'])
print()
print('Sample US-GAAP concepts:')
for i, concept in enumerate(list(data['facts']['us-gaap'].keys())[:10]):
    print(f'  {i+1}. {concept}')
print()

# Find revenue concept
if 'Revenues' in data['facts']['us-gaap']:
    rev = data['facts']['us-gaap']['Revenues']
    print('Revenue data available!')
    print(f'  Units: {list(rev["units"].keys())}')
    print(f'  Total data points: {len(rev["units"]["USD"])}')
    print(f'  Sample (latest 3):')
    for item in rev['units']['USD'][-3:]:
        print(f'    {item.get("end")}: ${item.get("val"):,} (Form: {item.get("form")})')
