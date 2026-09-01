from .US_Stocks import _residual_bins_63d as _us_stocks_bins_63d
from .US_Stocks import _residual_bins_21d as _us_stocks_bins_21d
from .US_Stocks import _residual_bins_5d as _us_stocks_bins_5d

_BALANCING_BINS_REGISTRY = {
	"us_stocks_63d": _us_stocks_bins_63d,
	"us_stocks_21d": _us_stocks_bins_21d,
	"us_stocks_5d": _us_stocks_bins_5d,
}


def resolve_balancing_bins(profile):
	if not isinstance(profile, str):
		raise TypeError(
			f"Unsupported balancing bins profile type: {type(profile).__name__}"
		)

	try:
		return _BALANCING_BINS_REGISTRY[profile]
	except KeyError as exc:
		keys = ", ".join(sorted(_BALANCING_BINS_REGISTRY))
		raise ValueError(
			f"Unknown balancing bins profile '{profile}'. Available profiles: {keys}"
		) from exc

