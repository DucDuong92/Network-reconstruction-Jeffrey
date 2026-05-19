"""Examples — run any product by changing the ProductConfig.

Run from the workspace root (where BACI19.csv, gravity19.csv, etc. live):
    python -m trade_jeffreys.examples
"""
from . import ProductConfig, run_product_analysis


# 1) BACI cocoa (HS 180500)
cocoa = ProductConfig(
    name="Cocoa (HS180500)",
    source="baci",
    trade_path="BACI19.csv",
    gravity_path="gravity19.csv",
    code_path="BACIcountry.csv",
    hs_codes=["180500"],
    output_prefix="BACI19_choco",
)

# 2) Uncom steel (HS 7208)
steel = ProductConfig(
    name="Steel (UN17)",
    source="uncom",
    trade_path="UNcom17_steel.csv",
    gravity_path="gravity17.csv",
)

# 3) Uncom fabric (HS 5208)
fabric = ProductConfig(
    name="Fabric (UN23)",
    source="uncom",
    trade_path="UNcom23_fabric.csv",
    gravity_path="gravity21.csv",
)


if __name__ == "__main__":
    # Just run cocoa as a smoke test (no plots, terse).
    run_product_analysis(cocoa, do_plots=False, verbose=True)
