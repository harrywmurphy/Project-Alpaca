# Risk checks with Alpaca calls monkeypatched -- no network, no keys needed.
import pytest

from risk import limits


class _Account:
    buying_power = "100000"
    equity = "100000"


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setattr(limits, "get_trading_client", lambda: _Client())
    monkeypatch.setattr(limits, "_latest_price", lambda s: 100.0)
    monkeypatch.setattr(limits, "_position_qty", lambda c, s: 0.0)
    monkeypatch.setattr(limits, "_portfolio_exposure", lambda c: 0.0)


class _Client:
    def get_account(self): return _Account()


def test_small_order_allowed():
    ok, _ = limits.check_order("SPY", "buy", 10)
    assert ok


def test_position_cap_blocks(monkeypatch):
    monkeypatch.setattr(limits, "_position_qty", lambda c, s: 45.0)
    ok, reason = limits.check_order("SPY", "buy", 10)  # 45 + 10 > 50
    assert not ok and "max_position_shares" in reason


def test_notional_cap_blocks(monkeypatch):
    monkeypatch.setattr(limits, "_latest_price", lambda s: 500.0)
    ok, reason = limits.check_order("SPY", "buy", 40)  # under 50 shares, but $20k > $15k
    assert not ok and "max_notional_per_order" in reason


def test_exposure_cap_blocks(monkeypatch):
    monkeypatch.setattr(limits, "_portfolio_exposure", lambda c: 59_000.0)
    ok, reason = limits.check_order("SPY", "buy", 20)  # 59k + 2k > 60% of 100k
    assert not ok and "exposure" in reason


def test_fails_closed_on_api_error(monkeypatch):
    def boom(): raise RuntimeError("API down")
    monkeypatch.setattr(limits, "get_trading_client", boom)
    ok, reason = limits.check_order("SPY", "buy", 10)
    assert not ok and "blocking" in reason
