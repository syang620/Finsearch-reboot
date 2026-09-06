"""Source-first annotation specification, authored without retrieval outputs.

Each named group is an independently inspected supporting passage (or table).
Literal anchors bind that judgment to every matching corpus occurrence. They
are not searches against any ranked retriever. See ANNOTATION.md for limits.
"""

# (document type, literal anchors). Text anchors are case-sensitive source text.
AAPL = {
    "sales": ("table", ["Services", "Total net sales", "{year}"]),
    "rd_table": ("table", ["Research and development", "{year}"]),
    "margin_pct": ("table", ["Gross margin percentage:", "Services"]),
    "margin_dollars": ("table", ["Gross margin:", "Services"]),
    "inventory": ("text_chunk", ["Inventories are measured using the first-in, first-out method."]),
    "cash": ("text_chunk", ["All highly liquid investments with maturities of three months or less at the date of purchase are treated as cash equivalents."]),
    "supply_risk": ("text_chunk", ["Because the Company currently obtains certain components from single or limited sources, the Company is subject to significant supply and pricing risks."]),
    "breach_risk": ("text_chunk", ["Losses or unauthorized access to or releases of confidential information, including personal information"]),
    "macro_risk": ("text_chunk", ["Adverse macroeconomic conditions, including"]),
    "background": ("text_chunk", ["The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories"]),
    "distribution": ("text_chunk", ["The Company also employs a variety of indirect distribution channels"]),
    "services_driver": ("text_chunk", ["Services net sales increased during {year}"]),
    "margin_driver": ("text_chunk", ["Services gross margin percentage increased during {year}"]),
    "seasonality": ("text_chunk", ["The Company has historically experienced higher net sales in its first quarter"]),
    "paper": ("text_chunk", ["The Company uses net proceeds from the commercial paper program for general corporate purposes"]),
    "cyber": ("text_chunk", ["The Company’s management, led by its Head of Corporate Information Security"]),
    "reports": ("text_chunk", ["The Company’s Annual Reports on Form 10-K, Quarterly Reports on Form 10-Q"]),
    "supply_business": ("text_chunk", ["The Company uses some custom components that are not commonly used by its competitors"]),
}
AMZN = {
    "sales": ("table", ["Net product sales", "Net service sales", "{year}"]),
    "segments": ("table", ["North America", "International", "AWS", "Net sales", "{year}"]),
    "property": ("table", ["Leased Square Footage", "Owned Square Footage", "AWS"]),
    "inventory": ("text_chunk", ["Inventories, consisting of products available for sale, are primarily accounted for using the first-in first-out method"]),
    "unearned": ("text_chunk", ["Unearned revenue is recorded when payments are received or due in advance of performing our service obligations"]),
    "shipping_risk": ("text_chunk", ["We rely on a limited number of shipping companies to deliver inventory to us and completed orders to our customers."]),
    "competition_risk": ("text_chunk", ["Our businesses are rapidly evolving and intensely competitive"]),
    "stock_risk": ("text_chunk", ["inventory markdowns or write-offs"]),
    "principles": ("text_chunk", ["We are guided by four principles: customer obsession rather than competitor focus"]),
    "sellers": ("text_chunk", ["We are not the seller of record in these transactions."]),
    "aws_driver": ("text_chunk", ["AWS sales increased", "in {year}", "customer usage"]),
    "na_driver": ("text_chunk", ["North America sales increased", "in {year}", "unit sales"]),
    "settlement": ("text_chunk", ["our receivables from consumers settle quickly"]),
    "advertising": ("text_chunk", ["Revenue is recognized as ads are delivered based on the number of clicks or impressions."]),
    "cyber": ("text_chunk", ["The Security Committee, which is comprised of independent directors"]),
    "workforce": ("text_chunk", ["full-time and part-time employees", "independent contractors and temporary personnel"]),
    "prime": ("text_chunk", ["Amazon Prime, a membership program that includes fast, free shipping"]),
    "fulfillment": ("text_chunk", ["We fulfill customer orders in a number of ways"]),
}
MSFT = {
    "revenue": ("table", ["Revenue", "Operating income", "{year}"]),
    "rd_table": ("table", ["Research and development", "{year}"]),
    "segments": ("table", ["Intelligent Cloud", "More Personal Computing", "Revenue", "{year}"]),
    "inventory": ("text_chunk", ["Inventories are stated at average cost, subject to the lower of cost or net realizable value."]),
    "cash": ("text_chunk", ["We consider all highly liquid interest-earning investments with a maturity of three months or less at the date of purchase to be cash equivalents."]),
    "cyber_risk": ("text_chunk", ["Cyberthreats are constantly evolving and becoming increasingly sophisticated and complex"]),
    "investment_risk": ("text_chunk", ["We make significant investments in products and services that may not achieve expected returns."]),
    "monetization_risk": ("text_chunk", ["We may not achieve significant revenue from new product, service, and distribution channel investments for several years, if at all."]),
    "cloud_scale": ("text_chunk", ["Our cloud business benefits from three economies of scale:"]),
    "linkedin": ("text_chunk", ["LinkedIn connects the world’s professionals to make them more productive and successful"]),
    "revenue_driver": ("text_chunk", ["Intelligent Cloud revenue increased driven by Azure."]),
    "rd_driver": ("text_chunk", ["Research and development expenses increased"]),
    "oem": ("text_chunk", ["The largest component of the OEM business is the Windows operating system pre-installed on devices."]),
    "multitenancy": ("text_chunk", ["multi-tenancy locations that lower application maintenance labor costs"]),
    "cyber": ("text_chunk", ["Our Board of Directors oversees cybersecurity risk."]),
    "windows": ("text_chunk", ["Windows faces competition from various software products and from alternative platforms and devices"]),
    "short_investments": ("text_chunk", ["original maturities of greater than three months and remaining maturities of less than one year"]),
}

# Strata are mutually exclusive primary labels; required_groups express distinct
# evidence needs. More than one alternative chunk may satisfy a group.
# Explicit grade-one groups support part of a query but not the requested fact.
# Explicit zero-grade groups are source-inspected within-filing hard negatives.
SPECS = {
"AAPL": [
 ("direct_fact", "What Services net sales are reported for {year} in Apple's category sales table?", ["sales"], [], []),
 ("direct_fact", "Find Apple's research and development expense for {year} in the annual expense table.", ["rd_table"], [], []),
 ("direct_fact", "What percentage gross margin did Apple report for Services in {year}?", ["margin_pct"], [], []),
 ("narrative", "Which inventory cost-flow method does Apple disclose in its {year} accounting policies?", ["inventory"], [], []),
 ("narrative", "What maturity limit defines Apple's cash equivalents in the {year} filing?", ["cash"], [], []),
 ("risk_factors", "In Apple's {year} risk factors, why can limited component suppliers create pricing and availability exposure?", ["supply_risk"], ["supply_business"], []),
 ("risk_factors", "What consequences does Apple's {year} risk discussion associate with disclosure or theft of confidential information?", ["breach_risk"], [], []),
 ("risk_factors", "How could weak economic conditions hurt consumer demand according to Apple's {year} risk factors?", ["macro_risk"], [], []),
 ("business_growth", "What products does Apple design and sell, and how does its {year} business overview define its fiscal year end?", ["background"], [], []),
 ("business_growth", "Which indirect sales channels does Apple describe in its {year} business overview?", ["distribution"], [], []),
 ("mda", "What drove the increase in Apple's Services net sales during {year}, according to management?", ["services_driver"], ["sales"], []),
 ("mda", "What explanation does Apple give for the change in Services gross margin percentage in {year}?", ["margin_driver"], ["margin_pct"], []),
 ("paraphrase", "Why does Apple's holiday shopping period make the opening quarter unusually strong in its {year} report?", ["seasonality"], [], []),
 ("paraphrase", "According to Apple's {year} report, what does it do with money raised through short-dated unsecured IOUs?", ["paper"], [], []),
 ("section_specific", "Find Item 1C of Apple's {year} 10-K: who leads management of material cyber threats?", ["cyber"], [], []),
 ("section_specific", "In Available Information, how does Apple make its {year} SEC reports accessible?", ["reports"], [], []),
 ("hard_negative", "In Apple's {year} filing, find the definition of cash-like investments, not its short-term borrowing program.", ["cash"], [], ["paper"]),
 ("hard_negative", "Locate Apple's {year} Services margin rate disclosure, not the table of gross profit dollars.", ["margin_pct"], [], ["margin_dollars"]),
 ("multi_evidence", "From Apple's {year} filing, find both the business description of custom component sourcing and the risk assessment of limited suppliers.", ["supply_business", "supply_risk"], [], []),
 ("multi_evidence", "Find Apple's {year} Services sales figure and management's explanation of its year-over-year growth.", ["sales", "services_driver"], [], []),
],
"AMZN": [
 ("direct_fact", "What are Amazon's {year} net product and net service sales in its consolidated statements?", ["sales"], [], []),
 ("direct_fact", "Locate Amazon's {year} segment net sales for North America, International and AWS.", ["segments"], [], []),
 ("direct_fact", "How much leased and owned AWS square footage does Amazon disclose in its {year} properties table?", ["property"], [], []),
 ("narrative", "How are Amazon's inventories costed and valued in the {year} accounting estimates disclosure?", ["inventory"], [], []),
 ("narrative", "When does Amazon record unearned revenue and over what period does it recognize it, in the {year} filing?", ["unearned"], [], []),
 ("risk_factors", "What delivery-network risks arise from dependence on a limited number of shipping companies in Amazon's {year} filing?", ["shipping_risk"], [], []),
 ("risk_factors", "How does Amazon characterize the intensity and breadth of competition in its {year} risk factors?", ["competition_risk"], [], []),
 ("risk_factors", "What can happen to profitability when Amazon stocks too much inventory, according to its {year} risks?", ["stock_risk"], [], []),
 ("business_growth", "What four operating principles guide Amazon according to its {year} business overview?", ["principles"], [], []),
 ("business_growth", "How does Amazon earn fees from sellers without being the seller of record, in its {year} business description?", ["sellers"], [], []),
 ("mda", "What explains AWS sales growth during {year} in Amazon's management discussion?", ["aws_driver"], ["segments"], []),
 ("mda", "What explains growth in North American sales during {year} in Amazon's management discussion?", ["na_driver"], ["segments"], []),
 ("paraphrase", "Why does Amazon get paid by shoppers quickly, as described in the working-capital overview of its {year} filing?", ["settlement"], [], []),
 ("paraphrase", "In Amazon's {year} accounts, when does showing or clicking a sponsored message turn into recognized sales?", ["advertising"], [], []),
 ("section_specific", "In Item 1C of Amazon's {year} filing, which independent-director committee oversees cyber safeguards?", ["cyber"], [], []),
 ("section_specific", "Find Human Capital in Amazon's {year} filing: how are permanent staff supplemented?", ["workforce"], [], []),
 ("hard_negative", "Find how Amazon recognizes advertising revenue in {year}, not how it records advance payments for subscriptions.", ["advertising"], [], ["unearned"]),
 ("hard_negative", "Find AWS's leased-versus-owned facility area in Amazon's {year} filing, not AWS segment sales.", ["property"], [], ["segments"]),
 ("multi_evidence", "Find both how Amazon fulfills orders and the shipping-carrier dependency risk described in its {year} filing.", ["fulfillment", "shipping_risk"], [], []),
 ("multi_evidence", "From Amazon's {year} report, find Prime's customer benefits and the accounting treatment of advance membership payments.", ["prime", "unearned"], [], []),
],
"MSFT": [
 ("direct_fact", "Locate Microsoft's reported revenue and operating income for fiscal {year}.", ["revenue"], [], []),
 ("direct_fact", "What research and development expense does Microsoft report for fiscal {year}?", ["rd_table"], [], []),
 ("direct_fact", "Locate Microsoft's fiscal {year} revenue for Intelligent Cloud and More Personal Computing.", ["segments"], [], []),
 ("narrative", "How does Microsoft value inventory according to its fiscal {year} accounting policies?", ["inventory"], [], []),
 ("narrative", "What is Microsoft's maturity-based definition of cash equivalents in fiscal {year}?", ["cash"], [], []),
 ("risk_factors", "Why are cyberattacks becoming harder to detect and defend against in Microsoft's {year} risk disclosure?", ["cyber_risk"], [], []),
 ("risk_factors", "What return-on-investment risk does Microsoft identify for new products and services in its {year} filing?", ["investment_risk"], [], []),
 ("risk_factors", "What delay or shortfall in monetization might Microsoft's new product investments face in its {year} risk discussion?", ["monetization_risk"], [], []),
 ("business_growth", "What three economies of scale support Microsoft's cloud business in its {year} business discussion?", ["cloud_scale"], [], []),
 ("business_growth", "What professional-network services and monetized offerings does Microsoft's {year} filing describe for LinkedIn?", ["linkedin"], [], []),
 ("mda", "What segment businesses explain Microsoft's overall revenue growth in fiscal {year}?", ["revenue_driver"], ["revenue"], []),
 ("mda", "What drove Microsoft's increase in research and development expenses in fiscal {year}?", ["rd_driver"], ["rd_table"], []),
 ("paraphrase", "In Microsoft's {year} distribution description, what software is chiefly sold already installed on new computers?", ["oem"], [], []),
 ("paraphrase", "Why does sharing one cloud application across customers reduce upkeep labor, according to Microsoft's {year} filing?", ["multitenancy"], [], []),
 ("section_specific", "Find Microsoft's {year} cybersecurity governance disclosure: how frequently is the board scheduled to review cyber risk?", ["cyber"], [], []),
 ("section_specific", "In the Windows competition discussion of Microsoft's {year} filing, what alternatives compete with the operating system?", ["windows"], [], []),
 ("hard_negative", "Find Microsoft's {year} inventory valuation policy, not its classification of cash-like investments.", ["inventory"], [], ["cash"]),
 ("hard_negative", "Find Microsoft's {year} data-center economies of scale, not the warning that new technology investments may disappoint.", ["cloud_scale"], [], ["investment_risk"]),
 ("multi_evidence", "From Microsoft's {year} report, find both board oversight of cyber risk and why evolving attacks are difficult to detect.", ["cyber", "cyber_risk"], [], []),
 ("multi_evidence", "Find Microsoft's fiscal {year} R&D spending figure and management's explanation for the increase.", ["rd_table", "rd_driver"], [], []),
]}

GROUPS = {"AAPL":AAPL, "AMZN":AMZN, "MSFT":MSFT}

# Source-year differences, established before any retrieval. The 2025 source
# no longer contains the 2024 inventory-policy paragraph; do not invent a label.
OVERRIDES = {
    ("MSFT", 2025, "inventory"): ("text_chunk", ["We believe use of a discounted cash flow approach is the most reliable indicator of the fair values of the businesses."]),
}
QUERY_OVERRIDES = {
    ("MSFT", 2025, 4): "What valuation approach does Microsoft use when testing goodwill for impairment in its fiscal 2025 disclosure?",
    ("MSFT", 2025, 17): "Find Microsoft's fiscal 2025 goodwill valuation approach, not its definition of cash equivalents.",
}
