"""Source-inspected semantic requirements, authored before any system answers.

Literal anchors only locate the selected source passages; they are not ranked
retrieval results. All anchors and year-specific requirements are inspected
against the source filings. Existing retrieval-benchmark labels are not edited.
"""

NAMES = {"AAPL": "Apple", "AMZN": "Amazon", "MSFT": "Microsoft"}

# Each entry: content kind, literal source anchors, independently authored claim.
PASSAGES = {
    "AAPL": {
        "policy": ("text_chunk", ["All highly liquid investments with maturities of three months or less at the date of purchase are treated as cash equivalents."],
                   "Apple treats highly liquid investments with maturities of three months or less at purchase as cash equivalents."),
        "growth": ("text_chunk", ["Services net sales increased during {year}", "advertising", "App Store", "cloud services"],
                   "Management attributes Services net-sales growth primarily to higher advertising, App Store and cloud-services sales."),
        "business": ("text_chunk", ["The Company uses some custom components that are not commonly used by its competitors", "available from only one source"],
                     "Apple uses custom components, and new products can depend on components available from only one source."),
        "risk": ("text_chunk", ["Because the Company currently obtains certain components from single or limited sources, the Company is subject to significant supply and pricing risks."],
                 "Single or limited component sources expose Apple to significant supply and pricing risks."),
        "attribution": ("text_chunk", ["Services gross margin percentage increased during {year}", "mix"],
                        "Management attributes the Services gross-margin percentage increase to a different services mix, not merely to higher net sales."),
        "hard_numeric": ("table", ["Gross margin percentage:", "Services"],
                         "Apple's Services gross-margin percentage is {value}%, not its Services gross-profit dollar amount."),
        "negative": ("table", ["Gross margin:", "Services"],
                     "Gross-profit dollars are not a margin percentage."),
    },
    "AMZN": {
        "policy": ("text_chunk", ["Unearned revenue is recorded when payments are received or due in advance of performing our service obligations and is recognized over the service period."],
                   "Amazon records unearned revenue for payments received or due before service obligations are performed, and recognizes revenue over the service period."),
        "growth": ("text_chunk", ["AWS sales increased", "in {year}", "The sales growth primarily reflects increased customer usage, partially offset by pricing changes"],
                   "AWS sales growth primarily reflects increased customer usage, partially offset by pricing changes driven primarily by long-term customer contracts."),
        "business": ("text_chunk", ["We fulfill customer orders in a number of ways", "digital delivery", "physical stores"],
                     "Amazon fulfills orders through operated North America/International networks, co-sourced or outsourced arrangements, digital delivery and physical stores."),
        "risk": ("text_chunk", ["We rely on a limited number of shipping companies to deliver inventory to us and completed orders to our customers.", "could negatively impact our operating results and customer experience"],
                 "Dependence on limited shipping companies creates risk: unacceptable terms or carrier performance/staffing difficulties can harm operating results and customer experience."),
        "attribution": ("text_chunk", ["AWS sales increased", "in {year}", "increased customer usage, partially offset by pricing changes"],
                        "AWS growth was driven primarily by customer usage; pricing changes were a partial offset rather than the stated positive growth driver."),
        "hard_numeric": ("table", ["Leased Square Footage", "Owned Square Footage", "AWS"],
                         "AWS leased facility area is {leased} and owned area is {owned}, in thousands of square feet, not segment sales dollars."),
        "negative": ("table", ["North America", "International", "AWS", "Net sales", "{year}"],
                     "AWS segment sales do not measure facility area."),
    },
    "MSFT": {
        "policy": ("text_chunk", ["We consider all highly liquid interest-earning investments with a maturity of three months or less at the date of purchase to be cash equivalents."],
                   "Microsoft treats highly liquid interest-earning investments with maturities of three months or less at purchase as cash equivalents."),
        "growth": ("text_chunk", ["Intelligent Cloud revenue increased driven by Azure.", "Productivity and Business Processes revenue increased", "More Personal Computing revenue increased"],
                   "Growth came across segments: Azure drove Intelligent Cloud; Office 365 Commercial drove Productivity and Business Processes; Gaming drove More Personal Computing."),
        "business": ("text_chunk", ["Our Board of Directors oversees cybersecurity risk.", "Cybersecurity reviews by the Board are scheduled to occur at least quarterly"],
                     "Microsoft's board oversees cybersecurity risk and schedules reviews at least quarterly, more often when necessary or advisable."),
        "risk": ("text_chunk", ["Cyberthreats are constantly evolving and becoming increasingly sophisticated and complex, increasing the difficulty of detecting and successfully defending against them."],
                 "Evolving, increasingly sophisticated and complex cyberthreats make detection and defense harder."),
        "attribution": ("text_chunk", ["Research and development expenses increased", "Gaming", "cloud engineering"],
                        "The R&D expense increase was driven by Gaming (including the Activision Blizzard acquisition) and cloud-engineering investments; management did not attribute it only to AI."),
        "hard_numeric": ("table", ["Research and development", "{year}"],
                         "Microsoft's R&D expense is {value} million USD, not its R&D tax-credit amount."),
        "negative": ("table", ["Research and development credit", "{year}"],
                     "The R&D tax credit is not R&D operating expense."),
    },
}

HARD_VALUES = {
    ("AAPL", 2024): {"value": 73.9}, ("AAPL", 2025): {"value": 75.4},
    ("AMZN", 2023): {"leased": 20434, "owned": 17770},
    ("AMZN", 2024): {"leased": 24875, "owned": 24052},
    ("MSFT", 2024): {"value": 29510}, ("MSFT", 2025): {"value": 32488},
}

OVERRIDES = {
    ("AAPL", 2025, "attribution"): (
        ["Services gross margin percentage increased during 2025", "different mix of services, partially offset by higher costs"],
        "Management attributes the Services gross-margin percentage increase primarily to a different services mix, partially offset by higher costs; higher sales alone explain a different measure."),
    ("MSFT", 2025, "growth"): (None,
        "Growth came across segments: Azure drove Intelligent Cloud; Microsoft 365 Commercial cloud drove Productivity and Business Processes; Gaming and Search/news advertising drove More Personal Computing."),
    ("MSFT", 2025, "attribution"): (
        ["Research and development expenses increased", "cloud and AI engineering and Gaming", "Activision Blizzard acquisition"],
        "The R&D expense increase was driven by cloud/AI engineering and Gaming, including the Activision Blizzard acquisition, not solely by AI."),
}

QUESTIONS = {
    "AAPL": {
        "policy": "Under Apple's FY{year} 10-K accounting policy, which investments qualify as cash equivalents?",
        "growth": "Report Apple's FY{year} total revenue in USD millions and explain management's stated drivers of Services sales growth.",
        "multi": "According to Apple's FY{year} 10-K, how does it source some custom components, and what supply/pricing exposure does that create?",
        "attribution": "In Apple's FY{year} 10-K, was the improvement in Services gross-margin percentage attributed to higher sales or to a change in mix? Include any stated offset.",
        "hard": "What Services gross-margin percentage did Apple report for FY{year}? Give the percentage, not the Services gross-profit dollar amount.",
    },
    "AMZN": {
        "policy": "Under Amazon's FY{year} 10-K revenue policy, when is unearned revenue recorded and when is it recognized as revenue?",
        "growth": "Report Amazon's FY{year} total revenue in USD millions and explain management's stated drivers and offsets for AWS sales growth.",
        "multi": "From Amazon's FY{year} 10-K, summarize its order-fulfillment methods and explain the risk from relying on a limited number of shipping companies.",
        "attribution": "Did Amazon's FY{year} 10-K attribute AWS sales growth to higher pricing rather than customer usage? Explain the stated direction of the pricing effect.",
        "hard": "In Amazon's FY{year} properties disclosure, what are AWS's leased and owned facility areas in thousands of square feet? Do not substitute AWS sales.",
    },
    "MSFT": {
        "policy": "Under Microsoft's FY{year} 10-K accounting policy, which investments qualify as cash equivalents?",
        "growth": "Report Microsoft's FY{year} total revenue in USD millions and identify the businesses management says drove growth in each of its three segments.",
        "multi": "According to Microsoft's FY{year} 10-K, who oversees cybersecurity risk and how often, and why are cyberthreats becoming harder to detect and defend against?",
        "attribution": "Does Microsoft's FY{year} 10-K attribute the increase in R&D expense solely to AI? Identify the other stated drivers and acquisition context.",
        "hard": "What R&D expense did Microsoft report for FY{year}, in USD millions? Use the operating-expense amount, not the R&D tax credit.",
    },
}
