"""XBRL -> canonical concept normalization.

US-GAAP gives the same economic quantity several tags depending on filer
preference and year (`Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`,
`OperatingIncomeLoss` vs a computed subtotal, ...). Downstream code must never
see that variety, so every ingested fact is mapped to a canonical concept here.

The alias list for each concept is ORDERED BY PREFERENCE: when a filer reports
several aliases for one period we keep the highest-priority one and record the
loser in the fact's `raw_concept` audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Statement = Literal["income", "balance", "cashflow", "other"]
PeriodType = Literal["duration", "instant"]


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    statement: Statement
    period_type: PeriodType
    label: str
    aliases: tuple[str, ...]
    sign: int = 1  # multiply raw value by this (e.g. capex reported positive)


CONCEPTS: tuple[ConceptSpec, ...] = (
    # ---------------- income statement ----------------
    ConceptSpec("revenue", "income", "duration", "Revenue", (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    )),
    ConceptSpec("cost_of_revenue", "income", "duration", "Cost of revenue", (
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
    )),
    ConceptSpec("gross_profit", "income", "duration", "Gross profit", ("GrossProfit",)),
    ConceptSpec("rd_expense", "income", "duration", "R&D expense", (
        "ResearchAndDevelopmentExpense",
    )),
    ConceptSpec("sga_expense", "income", "duration", "SG&A expense", (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    )),
    ConceptSpec("operating_income", "income", "duration", "Operating income", (
        "OperatingIncomeLoss",
    )),
    ConceptSpec("depreciation_amortization", "cashflow", "duration", "D&A", (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndOther",
    )),
    # Many large filers report depreciation and intangible amortization as two
    # separate tags and never emit a combined one. These feed a derived D&A.
    ConceptSpec("depreciation_only", "cashflow", "duration", "Depreciation", (
        "Depreciation", "DepreciationNonproduction",
    )),
    ConceptSpec("amortization_intangibles", "cashflow", "duration",
                "Amortization of intangibles", (
        "AmortizationOfIntangibleAssets",
        "AmortizationOfAcquisitionCosts",
    )),
    ConceptSpec("interest_expense", "income", "duration", "Interest expense", (
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
        "InterestExpenseBorrowings",
        "InterestExpenseNonoperating",
        "InterestExpenseOther",
        "InterestIncomeExpenseNet",
    )),
    ConceptSpec("pretax_income", "income", "duration", "Pre-tax income", (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    )),
    ConceptSpec("income_tax_expense", "income", "duration", "Income tax expense", (
        "IncomeTaxExpenseBenefit",
    )),
    ConceptSpec("net_income", "income", "duration", "Net income", (
        "NetIncomeLoss", "ProfitLoss",
    )),
    ConceptSpec("eps_diluted", "income", "duration", "Diluted EPS", (
        "EarningsPerShareDiluted",
    )),
    # ---------------- balance sheet ----------------
    ConceptSpec("cash_and_equivalents", "balance", "instant", "Cash & equivalents", (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    )),
    ConceptSpec("short_term_investments", "balance", "instant", "Short-term investments", (
        "ShortTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
    )),
    ConceptSpec("accounts_receivable", "balance", "instant", "Accounts receivable", (
        "AccountsReceivableNetCurrent", "ReceivablesNetCurrent",
    )),
    ConceptSpec("inventory", "balance", "instant", "Inventory", (
        "InventoryNet", "InventoryFinishedGoods",
    )),
    ConceptSpec("current_assets", "balance", "instant", "Total current assets", (
        "AssetsCurrent",
    )),
    ConceptSpec("total_assets", "balance", "instant", "Total assets", ("Assets",)),
    ConceptSpec("goodwill", "balance", "instant", "Goodwill", ("Goodwill",)),
    ConceptSpec("intangible_assets", "balance", "instant", "Intangibles", (
        "IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet",
    )),
    ConceptSpec("accounts_payable", "balance", "instant", "Accounts payable", (
        "AccountsPayableCurrent", "AccountsPayableTradeCurrent",
    )),
    ConceptSpec("current_liabilities", "balance", "instant", "Total current liabilities", (
        "LiabilitiesCurrent",
    )),
    ConceptSpec("short_term_debt", "balance", "instant", "Short-term debt", (
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "NotesPayableCurrent",
        "OtherShortTermBorrowings",
        "CommercialPaper",
    )),
    ConceptSpec("long_term_debt", "balance", "instant", "Long-term debt", (
        # Order matters: the *noncurrent* tag is unambiguous. `LongTermDebt` is
        # the total including the current portion at many filers, so it ranks
        # below it and is corrected for in statements.add_derived().
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
        "SeniorNotesNoncurrent",
        "OtherLongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "UnsecuredDebt",
        "LongTermDebt",
    )),
    #: Some filers tag one combined debt figure. When present it beats summing
    #: components, because it is the filer's own definition of total debt.
    ConceptSpec("total_debt_reported", "balance", "instant", "Total debt (as reported)", (
        "DebtLongtermAndShorttermCombinedAmount",
    )),
    ConceptSpec("operating_lease_liability", "balance", "instant", "Operating lease liability", (
        "OperatingLeaseLiabilityNoncurrent", "OperatingLeaseLiability",
    )),
    ConceptSpec("total_liabilities", "balance", "instant", "Total liabilities", ("Liabilities",)),
    ConceptSpec("total_equity", "balance", "instant", "Total equity", (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    )),
    ConceptSpec("retained_earnings", "balance", "instant", "Retained earnings", (
        "RetainedEarningsAccumulatedDeficit",
    )),
    # ---------------- cash flow ----------------
    ConceptSpec("operating_cash_flow", "cashflow", "duration", "Operating cash flow", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    )),
    ConceptSpec("capex", "cashflow", "duration", "Capital expenditure", (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    )),
    ConceptSpec("investing_cash_flow", "cashflow", "duration", "Investing cash flow", (
        "NetCashProvidedByUsedInInvestingActivities",
    )),
    ConceptSpec("financing_cash_flow", "cashflow", "duration", "Financing cash flow", (
        "NetCashProvidedByUsedInFinancingActivities",
    )),
    ConceptSpec("dividends_paid", "cashflow", "duration", "Dividends paid", (
        "PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
    )),
    ConceptSpec("share_repurchase", "cashflow", "duration", "Share repurchases", (
        "PaymentsForRepurchaseOfCommonStock",
    )),
)

BY_NAME: dict[str, ConceptSpec] = {c.name: c for c in CONCEPTS}

#: XBRL tag -> (canonical concept, priority). Lower priority number wins.
ALIAS_INDEX: dict[str, tuple[str, int]] = {}
for _spec in CONCEPTS:
    for _rank, _alias in enumerate(_spec.aliases):
        # First writer wins so a tag shared by two concepts keeps its primary home.
        ALIAS_INDEX.setdefault(_alias, (_spec.name, _rank))

#: Concepts that are computed rather than tagged. Kept separate so the ingest
#: layer never persists a derived value as if a filer had reported it.
DERIVED_CONCEPTS: dict[str, str] = {
    "total_debt": "short_term_debt + long_term_debt",
    "net_debt": "total_debt - cash_and_equivalents - short_term_investments",
    "ebitda": "operating_income + depreciation_amortization",
    "ebit": "operating_income",
    "free_cash_flow": "operating_cash_flow - capex",
    "working_capital": "current_assets - current_liabilities",
    "gross_profit_derived": "revenue - cost_of_revenue",
    "tangible_equity": "total_equity - goodwill - intangible_assets",
}


def resolve(tag: str) -> tuple[str, int] | None:
    """Map an XBRL tag to (canonical concept, alias priority)."""
    return ALIAS_INDEX.get(tag)


def spec(concept: str) -> ConceptSpec | None:
    return BY_NAME.get(concept)


def label_for(concept: str) -> str:
    s = BY_NAME.get(concept)
    if s:
        return s.label
    return concept.replace("_", " ").title()


def is_flow(concept: str) -> bool:
    """Flow (duration) concepts sum across quarters; stock (instant) ones do not."""
    s = BY_NAME.get(concept)
    if s is not None:
        return s.period_type == "duration"
    return concept in {"ebitda", "ebit", "free_cash_flow"}


#: Tags that report a total already including the current portion. When the
#: long-term figure comes from one of these, adding short-term debt on top
#: would double count the current maturities.
LONG_TERM_TAGS_INCLUDING_CURRENT = frozenset({
    "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
})
