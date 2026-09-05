"""Issuer definitions for the offline demo corpus.

Every company here is FICTIONAL. Names, CIKs, financials and filing text are
invented for demonstration and testing. Nothing in this file describes, or is
intended to resemble, any real issuer's reported results.
"""
from __future__ import annotations

from creditlens.ingest.fixtures import IssuerSpec, ramp, seasonal

INDUSTRIAL_SEASON = (0.95, 1.03, 1.00, 1.02)
SOFTWARE_SEASON = (0.96, 1.00, 1.01, 1.03)
RETAIL_SEASON = (0.86, 0.92, 0.95, 1.27)

NOVACORE = IssuerSpec(
    ticker="NVCR",
    name="Novacore Industries, Inc.",
    cik="9990000001",
    industry="Diversified industrial manufacturing",
    sic="3559",
    profile="deteriorating investment grade",
    revenue=seasonal(ramp(2380, 2455), INDUSTRIAL_SEASON),
    gross_margin=ramp(0.281, 0.224),
    sga_pct=ramp(0.118, 0.131),
    rd_pct=ramp(0.031, 0.028),
    da_pct=ramp(0.046, 0.052),
    total_debt=ramp(6800, 9250),
    short_term_share=0.14,
    debt_rate=ramp(0.041, 0.062),
    capex_pct=ramp(0.052, 0.061),
    wc_drag_pct=ramp(0.004, 0.017),
    opening_cash=1180,
    opening_sti=240,
    opening_equity=7400,
    opening_retained=5100,
    goodwill=3900,
    intangibles=1450,
    ppe_pct_revenue=2.35,
    dso_days=58,
    dio_days=96,
    dpo_days=54,
    dividend_payout=0.34,
    buyback_pct_ocf=0.10,
    themes={
        "business": [
            "{name} designs and manufactures precision motion control systems, industrial "
            "automation equipment and aftermarket components for customers in transportation, "
            "energy infrastructure and heavy equipment end markets. We operate 31 manufacturing "
            "facilities across North America, Europe and Asia and sell through a combination of "
            "direct sales channels and a distributor network covering approximately 60 countries.",
            "Our business is organised into two reportable segments. Motion Systems supplies "
            "actuation and drivetrain assemblies under multi-year platform agreements, and "
            "Industrial Solutions supplies automation cells, controls and recurring service "
            "contracts. Aftermarket and service revenue represented a meaningful and comparatively "
            "stable portion of consolidated revenue for the period, which partially offsets the "
            "cyclicality of original equipment demand.",
            "Demand for our products is closely tied to customer capital spending cycles. Order "
            "intake in the current period reflected continued softness in heavy equipment "
            "platforms, partially offset by resilience in energy infrastructure programmes.",
        ],
        "risk_intro": [
            "The following risk factors should be read together with our consolidated financial "
            "statements. Our leverage has increased materially over the past several periods, and "
            "total debt of {debt} against trailing EBITDA of {ebitda} leaves less headroom to "
            "absorb a further downturn than in prior years.",
        ],
        "risks": [
            "We have substantial indebtedness that could adversely affect our financial condition. "
            "As of the end of the period, total debt outstanding was {debt} and net debt was "
            "{net_debt}, representing net leverage of approximately {net_leverage}. Our credit "
            "agreement contains a maximum net leverage covenant and a minimum interest coverage "
            "covenant. While we were in compliance with all financial covenants at period end, "
            "continued margin compression would reduce headroom under those covenants, and a "
            "breach could permit lenders to accelerate the indebtedness.",
            "Rising interest rates have increased our cost of borrowing. A portion of our debt "
            "bears interest at floating rates, and approximately $2.1 billion of senior notes mature "
            "within the next 24 months. If we refinance those maturities at prevailing rates, "
            "interest expense would increase relative to the current run rate and interest coverage "
            "of {coverage} would decline further.",
            "Input cost inflation and an unfavourable shift in product mix have compressed gross "
            "margin. Steel, castings and electronic components have repriced faster than our "
            "ability to pass through cost under fixed-price platform agreements, several of which "
            "extend beyond the next fiscal year. We may not be able to renegotiate pricing on "
            "commercially acceptable terms.",
            "A downgrade of our credit ratings would increase our cost of capital and could reduce "
            "access to the commercial paper market on which we rely for seasonal working capital "
            "needs. Rating agencies have cited our leverage trajectory and free cash flow "
            "conversion as key considerations.",
            "We depend on a concentrated group of customers. Our five largest customers accounted "
            "for a substantial share of consolidated revenue, and the loss or insourcing of a major "
            "platform programme would have a material adverse effect on operating results and cash "
            "flow available for debt service.",
            "Our pension obligations and long-dated warranty exposures are sensitive to discount "
            "rates and claims experience. Adverse developments would increase required cash "
            "contributions at a time when free cash flow of {fcf} already provides limited surplus "
            "after capital expenditure and the dividend.",
        ],
        "results": [
            "Revenue for {period} was {revenue} on a trailing twelve month basis. Volume was "
            "broadly stable, with modest price realisation offset by weaker mix as aftermarket "
            "content declined as a share of the total.",
            "Gross margin declined year over year, driven principally by higher input costs, "
            "unfavourable absorption at two European facilities and the mix shift toward original "
            "equipment platforms. Operating margin of {operating_margin} reflects those pressures "
            "together with higher selling, general and administrative expense associated with "
            "systems implementation and elevated freight.",
            "EBITDA of {ebitda}, or {ebitda_margin} of revenue, decreased from the prior year "
            "period. Management continues to execute a cost reduction programme targeting footprint "
            "consolidation and indirect procurement, the benefits of which are expected to be "
            "realised over the following four to six quarters.",
        ],
        "liquidity": [
            "As of the end of {period}, we held cash, cash equivalents and short-term investments "
            "of {cash} and had total debt of {debt}. Our current ratio was {current_ratio}. We also "
            "maintain an undrawn $2.0 billion revolving credit facility that matures in three years "
            "and backstops our commercial paper programme.",
            "Cash provided by operating activities decreased relative to the prior year, reflecting "
            "lower earnings and a larger working capital build as inventory was positioned ahead of "
            "platform launches. Free cash flow of {fcf} on a trailing twelve month basis was applied "
            "to the dividend, modest share repurchases and scheduled debt service.",
            "We expect capital expenditure to remain elevated relative to historical levels as we "
            "complete capacity additions. Absent an improvement in operating cash flow, we expect "
            "to fund a portion of the upcoming maturities with new borrowings, which would keep "
            "leverage above our stated long-term target of below 3.0 times.",
        ],
        "market_risk": [
            "We are exposed to interest rate risk on our floating rate borrowings. A hypothetical "
            "100 basis point increase in short-term rates would increase annual interest expense by "
            "approximately $34 million based on the floating rate balances outstanding at period end.",
            "We are exposed to foreign currency risk, principally the euro and the Japanese yen, "
            "arising from manufacturing costs denominated in currencies other than the functional "
            "currency of the selling entity. We use forward contracts to hedge a portion of "
            "forecasted exposures for up to twelve months.",
            "Commodity price risk relates primarily to steel, aluminium and copper. We do not "
            "currently hedge a material portion of commodity exposure and instead seek to recover "
            "cost increases through contractual pass-through provisions, which typically operate "
            "with a two to three quarter lag.",
        ],
    },
)

ARAMONT = IssuerSpec(
    ticker="ARMT",
    name="Aramont Software Corporation",
    cik="9990000002",
    industry="Enterprise application software",
    sic="7372",
    profile="strong net-cash issuer",
    revenue=seasonal(ramp(1810, 2960), SOFTWARE_SEASON),
    gross_margin=ramp(0.806, 0.834),
    sga_pct=ramp(0.281, 0.252),
    rd_pct=ramp(0.169, 0.163),
    da_pct=ramp(0.041, 0.036),
    total_debt=[1500.0] * 8 + [1500.0, 1500.0, 1200.0, 1200.0, 1200.0, 1200.0],
    short_term_share=0.05,
    debt_rate=[0.0325] * 14,
    capex_pct=ramp(0.041, 0.036),
    wc_drag_pct=ramp(0.002, -0.004),
    opening_cash=5200,
    opening_sti=3100,
    opening_equity=9800,
    opening_retained=6400,
    goodwill=2600,
    intangibles=520,
    ppe_pct_revenue=0.72,
    dso_days=64,
    dio_days=0.5,
    dpo_days=28,
    dividend_payout=0.14,
    buyback_pct_ocf=0.42,
    themes={
        "business": [
            "{name} provides cloud-delivered enterprise planning, financial consolidation and data "
            "governance software to large and mid-sized organisations. Substantially all new "
            "bookings are subscription based, and subscription revenue represents the large "
            "majority of consolidated revenue.",
            "Our go-to-market model combines a direct enterprise sales force with a partner "
            "ecosystem of systems integrators. Net revenue retention remained above 110% for the "
            "period, reflecting seat expansion and module attach within the installed base.",
            "We compete with large diversified platform vendors and with point solutions. We "
            "differentiate on time-to-value, breadth of prebuilt integrations and the depth of our "
            "governance capabilities in regulated industries.",
        ],
        "risk_intro": [
            "Our financial position is characterised by a substantial net cash balance. Cash and "
            "short-term investments of {cash} exceed total debt of {debt}, and interest coverage of "
            "{coverage} provides significant headroom. The risks below are therefore weighted "
            "toward operating and competitive factors rather than financing risk.",
        ],
        "risks": [
            "Our growth depends on continued adoption of cloud planning software. A slowdown in "
            "enterprise software budgets, longer procurement cycles or increased scrutiny of "
            "discretionary technology spending could reduce new bookings and slow revenue growth "
            "from the {revenue} reported on a trailing twelve month basis.",
            "We face intense competition from larger vendors that bundle competing functionality "
            "into broader suites and may price aggressively. Competitive pressure could reduce our "
            "gross margin, which was supported by favourable hosting economics during the period.",
            "A security incident affecting customer data would damage our reputation and could "
            "result in significant liability. We process sensitive financial data on behalf of "
            "customers and are subject to evolving data protection requirements across multiple "
            "jurisdictions.",
            "We have historically returned a substantial portion of operating cash flow to "
            "shareholders through repurchases. Although free cash flow of {fcf} comfortably funds "
            "the current programme, a decision to pursue a large acquisition could change our "
            "capital structure and reduce the net cash position that currently supports our credit "
            "profile.",
            "Our results depend on retaining engineering and go-to-market talent in competitive "
            "labour markets. Increased compensation costs would pressure operating margin, which "
            "was {operating_margin} for the period.",
        ],
        "results": [
            "Total revenue for {period} was {revenue} on a trailing twelve month basis, with growth "
            "driven by subscription expansion within the installed base and continued new logo "
            "additions in regulated industries.",
            "Gross margin expanded year over year as hosting efficiency improvements and a higher "
            "mix of subscription revenue more than offset increased customer support investment. "
            "Operating margin of {operating_margin} reflects disciplined hiring and lower "
            "sales and marketing intensity as a percentage of revenue.",
            "EBITDA of {ebitda} represented {ebitda_margin} of revenue. Research and development "
            "spending remained a priority and increased in absolute terms while declining modestly "
            "as a share of revenue.",
        ],
        "liquidity": [
            "We ended {period} with cash, cash equivalents and short-term investments of {cash} "
            "against total debt of {debt}, a net cash position. Our current ratio was "
            "{current_ratio}. Our $2.5 billion revolving credit facility was undrawn.",
            "Cash provided by operating activities benefited from favourable working capital as "
            "deferred revenue grew with billings. Free cash flow was {fcf} on a trailing twelve "
            "month basis, and free cash flow conversion from EBITDA remained high.",
            "We repaid $300 million of maturing senior notes during the period from cash on hand "
            "and did not refinance the maturity. We expect to continue funding capital returns from "
            "free cash flow rather than from incremental borrowing.",
        ],
        "market_risk": [
            "Our debt bears interest at fixed rates, so a change in market interest rates would not "
            "have a material effect on interest expense. Interest income on our investment "
            "portfolio is sensitive to rates; a hypothetical 100 basis point decline would reduce "
            "annual interest income by approximately $78 million.",
            "Our investment portfolio consists principally of government and high-grade corporate "
            "securities with a weighted average duration under eighteen months, limiting mark to "
            "market exposure.",
            "Approximately one third of revenue is denominated in currencies other than the US "
            "dollar, principally the euro and pound sterling. We hedge a portion of forecast "
            "exposures with forward contracts.",
        ],
    },
)

KESTRA = IssuerSpec(
    ticker="KSTR",
    name="Kestra Systems Holdings plc",
    cik="9990000003",
    industry="Infrastructure software and data platforms",
    sic="7372",
    profile="leveraged acquirer deleveraging",
    revenue=seasonal(ramp(3210, 4160), SOFTWARE_SEASON),
    gross_margin=ramp(0.712, 0.761),
    sga_pct=ramp(0.242, 0.219),
    rd_pct=ramp(0.152, 0.148),
    da_pct=ramp(0.098, 0.079),
    total_debt=ramp(28400, 20900),
    short_term_share=0.09,
    debt_rate=ramp(0.055, 0.051),
    capex_pct=ramp(0.062, 0.055),
    wc_drag_pct=ramp(0.006, 0.001),
    opening_cash=4100,
    opening_sti=900,
    opening_equity=6200,
    opening_retained=2100,
    goodwill=26500,
    intangibles=9800,
    ppe_pct_revenue=1.55,
    dso_days=71,
    dio_days=1.0,
    dpo_days=33,
    dividend_payout=0.22,
    buyback_pct_ocf=0.06,
    themes={
        "business": [
            "{name} provides database, middleware and data platform software together with managed "
            "cloud infrastructure services. Following the acquisition of a large healthcare data "
            "platform business, we operate two reportable segments: Platform Software and Cloud "
            "Infrastructure Services.",
            "A substantial portion of revenue is recurring, comprising license support renewals and "
            "cloud subscription contracts. Support renewal rates have remained high, which provides "
            "considerable visibility into near-term cash generation.",
            "Our strategy is to migrate the installed license base to cloud subscriptions while "
            "applying free cash flow to debt reduction. Management has stated a public objective of "
            "returning net leverage below 3.0 times.",
        ],
        "risk_intro": [
            "Our capital structure reflects debt incurred to fund a large acquisition. Total debt "
            "of {debt} against trailing EBITDA of {ebitda} corresponds to gross leverage of "
            "{leverage}. Although we have reduced debt in each of the last several quarters, our "
            "leverage remains elevated relative to peers.",
        ],
        "risks": [
            "Our substantial indebtedness limits financial flexibility. Total debt was {debt} at "
            "period end with net debt of {net_debt}. A significant portion of operating cash flow "
            "is dedicated to debt service, reducing funds available for acquisitions, capital "
            "expenditure and shareholder returns.",
            "We may not achieve the cost synergies and cross-sell benefits assumed at the time of "
            "the acquisition. Integration of billing, support and cloud operations is ongoing, and "
            "shortfalls would slow the deleveraging path and could delay achievement of our stated "
            "net leverage objective.",
            "A meaningful portion of goodwill and acquired intangible assets is recorded on our "
            "balance sheet. Sustained underperformance in the acquired business could result in an "
            "impairment charge that, while non-cash, would reduce reported equity and tighten "
            "covenant headroom under agreements that reference net worth.",
            "Migration of customers from perpetual license support to cloud subscriptions may "
            "temporarily reduce revenue and margin, because subscription revenue is recognised "
            "over time whereas support renewals bill annually in advance. Interest coverage of "
            "{coverage} would be pressured by a faster than expected transition.",
            "We face concentration in large enterprise and public sector customers with extended "
            "procurement cycles. Budget deferrals would reduce bookings and slow the growth in free "
            "cash flow of {fcf} on which our deleveraging plan depends.",
        ],
        "results": [
            "Revenue for {period} was {revenue} on a trailing twelve month basis, with cloud "
            "subscription growth more than offsetting the planned decline in perpetual license "
            "revenue.",
            "Gross margin improved as the acquired infrastructure footprint was consolidated and "
            "utilisation increased. Operating margin of {operating_margin} benefited from lower "
            "acquisition-related amortisation and continued integration savings.",
            "EBITDA of {ebitda}, or {ebitda_margin} of revenue, increased year over year. "
            "Management expects further margin expansion as duplicate facilities are exited.",
        ],
        "liquidity": [
            "We ended {period} with cash and short-term investments of {cash} and total debt of "
            "{debt}. Our current ratio was {current_ratio}. We repaid debt during the period from "
            "operating cash flow and expect to continue applying free cash flow primarily to debt "
            "reduction until our net leverage objective is achieved.",
            "Free cash flow was {fcf} on a trailing twelve month basis. Our nearest significant "
            "maturity is a $3.5 billion term loan tranche due in eighteen months, which we expect "
            "to repay or refinance well in advance.",
            "Interest expense remains elevated but has begun to decline in line with lower average "
            "debt balances. Interest coverage was {coverage} for the trailing twelve month period, "
            "compared with a covenant minimum of 2.5 times.",
        ],
        "market_risk": [
            "Approximately 40% of our debt bears interest at floating rates. A hypothetical 100 "
            "basis point increase in reference rates would increase annual interest expense by "
            "approximately $84 million, before the effect of interest rate swaps.",
            "We use interest rate swaps to fix a portion of floating rate exposure. The notional "
            "amount outstanding at period end was $4.0 billion with a weighted average remaining "
            "term of two years.",
            "We are exposed to foreign currency translation risk across European and Asian "
            "operations. We do not hedge translation exposure and report the effect within other "
            "comprehensive income.",
        ],
    },
)

HARBRIDGE = IssuerSpec(
    ticker="HRBG",
    name="Harbridge Retail Group, Inc.",
    cik="9990000004",
    industry="Specialty retail",
    sic="5651",
    profile="stressed, negative free cash flow",
    revenue=seasonal(ramp(3620, 3180), RETAIL_SEASON),
    gross_margin=ramp(0.341, 0.276),
    sga_pct=ramp(0.276, 0.301),
    rd_pct=[0.0] * 14,
    da_pct=ramp(0.038, 0.047),
    total_debt=ramp(4180, 5460),
    short_term_share=0.22,
    debt_rate=ramp(0.058, 0.091),
    capex_pct=ramp(0.041, 0.026),
    wc_drag_pct=ramp(0.009, 0.024),
    opening_cash=620,
    opening_sti=0,
    opening_equity=2350,
    opening_retained=1400,
    goodwill=980,
    intangibles=310,
    ppe_pct_revenue=1.42,
    dso_days=9,
    dio_days=118,
    dpo_days=61,
    dividend_payout=0.0,
    buyback_pct_ocf=0.0,
    themes={
        "business": [
            "{name} operates specialty apparel and home goods stores across North America together "
            "with a direct-to-consumer digital channel. We operated approximately 1,180 stores at "
            "period end, down from prior year following planned closures.",
            "Our results are highly seasonal, with a disproportionate share of revenue and "
            "substantially all of our operating income generated in the fourth fiscal quarter. We "
            "borrow seasonally under an asset-based revolving credit facility to fund inventory "
            "purchases ahead of the holiday period.",
            "We have undertaken a transformation programme comprising store fleet rationalisation, "
            "supply chain consolidation and a rebuild of our digital platform.",
        ],
        "risk_intro": [
            "Our credit profile has weakened. Total debt of {debt} against trailing EBITDA of "
            "{ebitda} implies gross leverage of {leverage}, free cash flow was {fcf}, and "
            "liquidity of {cash} provides limited cushion against a weak holiday season.",
        ],
        "risks": [
            "We have generated negative free cash flow in recent periods. Free cash flow of {fcf} "
            "on a trailing twelve month basis, together with cash and equivalents of {cash}, means "
            "we depend on our asset-based revolving credit facility to fund seasonal working "
            "capital. Availability under that facility fluctuates with the borrowing base, which is "
            "determined by eligible inventory and receivables and is tested monthly.",
            "Our asset-based facility includes a springing fixed charge coverage covenant that is "
            "tested when excess availability falls below a specified threshold. A weak holiday "
            "season could reduce availability, trigger the covenant test and restrict access to the "
            "facility at the point of greatest need.",
            "Sustained gross margin erosion from elevated promotional activity and clearance of "
            "aged inventory has reduced earnings. Inventory turns have slowed, increasing markdown "
            "risk and consuming working capital.",
            "We face the risk that vendors tighten trade credit terms or require letters of credit "
            "or cash in advance. Any material reduction in trade payable support would accelerate "
            "cash usage and further pressure liquidity.",
            "A downgrade of our credit ratings, or negative commentary regarding our liquidity, "
            "could cause vendors and factoring counterparties to tighten terms, and could increase "
            "the cost of the incremental borrowing on which we currently rely. Interest coverage of "
            "{coverage} provides limited headroom.",
            "Our store leases represent substantial fixed obligations. Store closures generally "
            "require negotiated lease terminations, and the cash cost of exiting underperforming "
            "locations may exceed the operating losses avoided in the near term.",
        ],
        "results": [
            "Revenue for {period} was {revenue} on a trailing twelve month basis, reflecting lower "
            "comparable store sales and the effect of net store closures.",
            "Gross margin declined significantly, driven by elevated markdowns to clear aged "
            "inventory, deleverage of occupancy costs on lower sales and higher freight. Operating "
            "margin of {operating_margin} reflects those pressures together with selling, general "
            "and administrative expense that did not decline in line with sales.",
            "EBITDA was {ebitda}, or {ebitda_margin} of revenue. Management is executing cost "
            "reductions across corporate overhead and store labour, although a portion of the "
            "benefit is expected to be reinvested in price.",
        ],
        "liquidity": [
            "At the end of {period} we held cash and equivalents of {cash} and had total debt of "
            "{debt}, of which a significant portion was drawn under our asset-based revolving "
            "credit facility. Our current ratio was {current_ratio}, and a substantial portion of "
            "current assets consists of inventory that would be realised at a discount in a "
            "liquidation scenario.",
            "Cash used in operating activities reflected the inventory build ahead of the holiday "
            "season and lower earnings. Free cash flow of {fcf} on a trailing twelve month basis "
            "was funded through incremental borrowing under the revolving facility.",
            "We have suspended the dividend and share repurchases and reduced planned capital "
            "expenditure to preserve liquidity. We expect capital expenditure to remain below "
            "historical levels, which may over time impair the competitiveness of the store fleet.",
        ],
        "market_risk": [
            "Borrowings under our asset-based revolving credit facility bear interest at floating "
            "rates. A hypothetical 100 basis point increase in reference rates would increase "
            "annual interest expense by approximately $19 million based on average borrowings "
            "during the period.",
            "We source a majority of merchandise from suppliers in Asia, with substantially all "
            "purchases denominated in US dollars. Changes in the relative value of supplier "
            "currencies affect negotiated cost over time.",
            "We are exposed to commodity and freight cost volatility, particularly ocean freight "
            "and cotton. We do not use derivative instruments to hedge these exposures.",
        ],
    },
)

ISSUERS: tuple[IssuerSpec, ...] = (NOVACORE, ARAMONT, KESTRA, HARBRIDGE)
ISSUERS_BY_TICKER = {spec.ticker: spec for spec in ISSUERS}
