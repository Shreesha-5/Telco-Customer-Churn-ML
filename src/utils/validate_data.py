
import great_expectations as ge
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    """

    print("🔍 Starting data validation with Great Expectations...")

    # === 🔥 DATA CLEANING (CRITICAL FIX) ===

    # Convert TotalCharges to numeric (handles blanks like " ")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert other numeric columns safely
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # Handle missing values (better to use median instead of 0)
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["tenure"].fillna(df["tenure"].median(), inplace=True)
    df["MonthlyCharges"].fillna(df["MonthlyCharges"].median(), inplace=True)

    # === CREATE GREAT EXPECTATIONS DATASET ===
    ge_df = ge.dataset.PandasDataset(df)

    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")

    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")

    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # === NUMERIC VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    ge_df.expect_column_values_to_not_be_null("TotalCharges")

    # === DATA CONSISTENCY CHECK ===
    print("   🔗 Validating data consistency...")

    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # === RUN VALIDATION ===
    print("   ⚙️ Running complete validation suite...")
    results = ge_df.validate()

    # === PROCESS RESULTS ===
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations
